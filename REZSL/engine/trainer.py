import torch
import numpy as np

import torch.distributed as dist
from REZSL.utils.comm import *
from .inferencer import eval_zs_gzsl
from REZSL.modeling import weighted_RegressLoss,ADLoss, CPTLoss, build_zsl_pipeline, computeCoefficient, recordError, get_attributes_info, get_attr_group

def do_train(
        model,
        ReZSL,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        use_REZSL,
        RegNorm,
        RegType,
        scale,
        device,
        max_epoch,
        model_file_path,
        cfg
    ):

    best_performance = [-0.1, -0.1, -0.1, -0.1, -0.1] # ZSL, S, U, H, AUSUC
    best_epoch = -1
    att_all = res['att_all'].to(device)
    att_all_var = torch.var(att_all,dim=0)
    att_all_std = torch.sqrt(att_all_var+1e-12)
    print(att_all_std)
    att_seen = res['att_seen'].to(device)
    support_att_seen=att_seen

    print("-----use "+ RegType + " -----")
    Reg_loss = weighted_RegressLoss(RegNorm, RegType, device)
    CLS_loss = torch.nn.CrossEntropyLoss()

    losses = []
    cls_losses = []
    reg_losses = []

    model.train()

    for epoch in range(0, max_epoch):
        print("lr: %.8f"%(optimizer.param_groups[0]["lr"]))

        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []

        scheduler.step()

        num_steps = len(tr_dataloader)
        model_type = cfg.MODEL.META_ARCHITECTURE

        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            if iteration%50==0:
                index = torch.argmax(ReZSL.running_weights_Matrix)
                att_dim = batch_att.shape[1]
                d1 = index // att_dim
                d2 = index % att_dim
                print('index: (%d, %d), max weight: %.4f, corresponding offset: %.4f, max offset: %.4f'%(d1, d2, ReZSL.running_weights_Matrix[d1][d2], ReZSL.running_offset_Matrix[d1][d2], torch.max(ReZSL.running_offset_Matrix)))

            if model_type =="BasicNet" or model_type =="AttentionNet":
                v2s = model(x=batch_img, support_att=support_att_seen)

                if use_REZSL:
                    n = v2s.shape[0]
                    ReZSL.updateWeightsMatrix(v2s.detach(), batch_att.detach(), batch_label.detach())
                    weights = ReZSL.getWeights(n, att_dim, batch_label.detach()).detach()  # weights matrix does not need gradients
                else:
                    weights = None

                if model.module == None:
                    score, cos_dist = model.cosine_dis(pred_att=v2s, support_att=support_att_seen)
                else:
                    score, cos_dist = model.module.cosine_dis(pred_att=v2s, support_att=support_att_seen)

                Lreg = Reg_loss(v2s, batch_att, weights)
                Lcls = CLS_loss(score, batch_label)

                loss = lamd[0] * Lcls + lamd[1] * Lreg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_info = 'epoch: %d, it: %d/%d  |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, lr: %.10f' % \
                           (epoch + 1, iteration, num_steps, loss, Lcls, Lreg, optimizer.param_groups[0]["lr"])
                print(log_info)

            if model_type =="GEMNet":
                v2s, atten_v2s, atten_map, query = model(x=batch_img, support_att=support_att_seen)

                if use_REZSL:
                    n = v2s.shape[0]
                    ReZSL.updateWeightsMatrix(v2s.detach(), batch_att.detach(), batch_label.detach()) # or updateWeightsMatrix_inBatch
                    weights = ReZSL.getWeights(n, att_dim, batch_label.detach()).detach()  # weights matrix does not need gradients
                else:
                    weights = None

                if model.module == None:
                    score, cos_dist = model.cosine_dis(pred_att=v2s, support_att=support_att_seen)
                else:
                    score, cos_dist = model.module.cosine_dis(pred_att=v2s, support_att=support_att_seen)
                Lreg = Reg_loss(v2s, batch_att, weights)
                Lcls = CLS_loss(score, batch_label)

                attr_group = get_attr_group(cfg.DATASETS.NAME)

                Lad = ADLoss(query, attr_group)

                Lcpt = CPTLoss(atten_map, device)

                loss = lamd[0] * Lcls + lamd[1] * Lreg + lamd[2] * Lad + lamd[3] * Lcpt

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_info = 'epoch: %d, it: %d/%d  |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, ad_loss: %.4f, cpt_loss: %.4f  lr: %.10f' % \
                           (epoch + 1, iteration, num_steps, loss, Lcls, Lreg, Lad, Lcpt, optimizer.param_groups[0]["lr"])
                print(log_info)

            loss_epoch.append(loss.item())
            cls_loss_epoch.append(Lcls.item())
            reg_loss_epoch.append(Lreg.item())

        if is_main_process():
            losses += loss_epoch
            cls_losses += cls_loss_epoch
            reg_losses += reg_loss_epoch

            loss_epoch_mean = sum(loss_epoch)/len(loss_epoch)
            cls_loss_epoch_mean = sum(cls_loss_epoch)/len(cls_loss_epoch)
            reg_loss_epoch_mean = sum(reg_loss_epoch)/len(reg_loss_epoch)
            log_info = 'epoch: %d |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, lr: %.10f' % \
                       (epoch + 1, loss_epoch_mean, cls_loss_epoch_mean, reg_loss_epoch_mean, optimizer.param_groups[0]["lr"])
            print(log_info)
        mask = torch.gt(ReZSL.mean_value, 0.0)
        mean = torch.mean(torch.masked_select(ReZSL.mean_value, mask))
        std = torch.std(torch.masked_select(ReZSL.mean_value, mask))
        print('Train_mean_offset mean: ' + str(mean.item()) + '. std: ' + str(std.item()) + '.')

        synchronize()
        print('Current running_weights_Matrix: ')
        print(ReZSL.running_weights_Matrix)
        acc_seen, acc_novel, H, acc_zs, AUSUC, best_gamma = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            ReZSL,
            device)

        synchronize()


        if is_main_process():
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f, best_gamma=%.4f' % (acc_zs, acc_seen, acc_novel, H, AUSUC, best_gamma))

            if acc_zs > best_performance[0]:
                best_performance[0] = acc_zs

            if H > best_performance[3]:
                best_epoch=epoch+1
                best_performance[1:4] = [acc_seen, acc_novel, H]
                data = {}
                data["model"] = model.state_dict()
                torch.save(data, model_file_path)
                print('save best model: ' + model_file_path)

            if AUSUC > best_performance[4]:
                best_performance[4] = AUSUC
                model_file_path_AUSUC = model_file_path.split('.pth')[0]+'_AUSUC'+'.pth'
                torch.save(data, model_file_path_AUSUC)
                print('save best AUSUC model: ' + model_file_path_AUSUC)
            print("best: ep: %d" % best_epoch)
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f' % tuple(best_performance))

    if is_main_process():
        print("best: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f' % tuple(best_performance))

