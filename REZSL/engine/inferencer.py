import torch
import numpy as np
from sklearn.metrics import accuracy_score


def stack_cal(scores,bias):
    if not scores.shape[1] == bias.shape[1]:
        bias = bias.repeat(1, len(range(torch.cuda.device_count())))
    scores = scores - bias
    return scores

def inference(model,dataloadr,support_att,ReZSL,device):
    scores = []
    labels = []

    for iteration, (img, label_att, label) in enumerate(dataloadr):
        img = img.to(device)
        label = label.to(device)
        label_att = label_att.to(device)
        v2s = model(x=img, support_att=support_att, )
        if model.module == None:
            score, cos_dist = model.cosine_dis(pred_att=v2s, support_att=support_att)
        else:
            score, cos_dist = model.module.cosine_dis(pred_att=v2s, support_att=support_att)
        _, pred = score.max(dim=1)
        scores.append(score)
        labels.append(label)
        ReZSL.arrangeTestOffset(v2s.detach(), label_att.detach(), label.detach())


    return scores, labels

def get_accuracy(scores, labels, test_id):
    _, pred = scores.max(dim=1)
    pred = pred.view(-1).cpu()

    outpred = test_id[pred]
    outpred = np.array(outpred, dtype='int')

    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(labels == l)[0]
        acc += accuracy_score(labels[idx], outpred[idx])
    acc = acc / unique_labels.shape[0]

    return acc

def cal_AUSUC(model, ts_dataloadr, tu_dataloadr, support_att, cls_seen_num, cls_unseen_num, test_id, ReZSL, device):

    start_ts_scores, ts_labels = inference(model,ts_dataloadr,support_att,ReZSL,device)
    start_ts_scores, ts_labels = torch.cat(start_ts_scores, dim=0), torch.cat(ts_labels, dim=0)

    start_tu_scores, tu_labels = inference(model, tu_dataloadr, support_att, ReZSL, device)
    start_tu_scores, tu_labels = torch.cat(start_tu_scores, dim=0), torch.cat(tu_labels, dim=0)

    TestOffsetMatrix, Test_mean_value = ReZSL.averageTestOffset()
    mask = torch.gt(Test_mean_value, 0.0)
    mean = torch.mean(torch.masked_select(Test_mean_value, mask))
    std = torch.std(torch.masked_select(Test_mean_value, mask))

    print('Test_mean_offset mean: ' + str(mean.item()) + '. std: ' + str(std.item()) + '.')
    r = computePearson(support_att, ReZSL.test_cls_offset_mean, dim=0)
    print("Pearson Co-relation:")
    print(r.mean() * 100)
    ReZSL.afterTest()

    AUSUC_vector = []

    # get directly stack

    start_ts_acc = get_accuracy(start_ts_scores, ts_labels, test_id)
    start_tu_acc = get_accuracy(start_tu_scores, tu_labels, test_id)
    start_H = 2 * start_ts_acc * start_tu_acc / (start_ts_acc + start_tu_acc)

    best_H, best_ts, best_tu, best_gamma = start_H, start_ts_acc, start_tu_acc, 0.0

    AUSUC_vector.append((start_ts_acc, start_tu_acc))

    # start to bias to seen
    if start_tu_acc > 1e-12:
        test_gamma = 0.0
        test_gamma_interval = -0.01

        while True:
            test_gamma = test_gamma + test_gamma_interval
            bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma)
            bias_u = torch.zeros((1, cls_unseen_num))
            bias = torch.cat([bias_s, bias_u], dim=1).to(device)
            ts_scores, tu_scores = stack_cal(start_ts_scores, bias), stack_cal(start_tu_scores, bias)
            ts_acc, tu_acc = get_accuracy(ts_scores, ts_labels, test_id), get_accuracy(tu_scores, tu_labels, test_id)
            AUSUC_vector.append((ts_acc, tu_acc))
            H = 2 * ts_acc * tu_acc / (ts_acc + tu_acc)

            if H > best_H:
                best_H, best_ts, best_tu, best_gamma = H, ts_acc, tu_acc, test_gamma
            if not tu_acc > 1e-12:
                break

    # start to bias to unseen
    if start_ts_acc > 1e-12:
        test_gamma = 0.0
        test_gamma_interval = 0.01

        while True:
            test_gamma = test_gamma + test_gamma_interval
            bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma)
            bias_u = torch.zeros((1, cls_unseen_num))
            bias = torch.cat([bias_s, bias_u], dim=1).to(device)
            ts_scores, tu_scores = stack_cal(start_ts_scores, bias), stack_cal(start_tu_scores, bias)
            ts_acc, tu_acc = get_accuracy(ts_scores, ts_labels, test_id), get_accuracy(tu_scores, tu_labels, test_id)
            AUSUC_vector.append((ts_acc, tu_acc))
            H = 2 * ts_acc * tu_acc / (ts_acc + tu_acc)

            if H > best_H:
                best_H, best_ts, best_tu, best_gamma = H, ts_acc, tu_acc, test_gamma
            if not ts_acc > 1e-12:
                break

    # compute AUSUC
    sorted_AUSUC_vector = sorted(AUSUC_vector, key=lambda acc:acc[1])
    AUSUC = 0.0
    for i in range(len(sorted_AUSUC_vector)):
        if i+1 < len(sorted_AUSUC_vector):
            y, x = sorted_AUSUC_vector[i]
            next_y, next_x = sorted_AUSUC_vector[i+1]
            AUSUC = AUSUC+(next_x-x)*(y+next_y)*0.5

    return AUSUC, AUSUC_vector, start_H, start_ts_acc, start_tu_acc, best_H, best_ts, best_tu, best_gamma

def cal_accuracy(model, dataloadr, support_att, test_id, ReZSL, device, bias=None):
    scores, labels = inference(model, dataloadr, support_att, ReZSL, device)

    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    if bias is not None:
        scores = stack_cal(scores,bias)

    acc = get_accuracy(scores, labels, test_id)

    return acc

def eval(
        tu_loader,
        ts_loader,
        att_unseen,
        att_seen,
        cls_unseen_num,
        cls_seen_num,
        test_id,
        train_test_id,
        model,
        test_gamma,
        ReZSL,
        device
):
    print("test ZSL")
    support_att_unseen = att_unseen.repeat(len(range(torch.cuda.device_count())), 1)
    acc_zsl = cal_accuracy(model=model, dataloadr=tu_loader, support_att=support_att_unseen, test_id=test_id, ReZSL=ReZSL, device=device, bias=None)


    att = torch.cat((att_seen, att_unseen), dim=0)
    support_att = att

    print("test AUSUC")
    AUSUC, AUSUC_vector, start_H, start_ts_acc, start_tu_acc, best_H, best_ts, best_tu, best_gamma = cal_AUSUC(model, ts_loader, tu_loader, support_att, cls_seen_num, cls_unseen_num, train_test_id, ReZSL, device)

    H, acc_gzsl_seen, acc_gzsl_unseen = best_H, best_ts, best_tu

    return acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H, AUSUC, best_gamma

def eval_zs_gzsl(
        tu_loader,
        ts_loader,
        res,
        model,
        test_gamma,
        ReZSL,
        device
):
    model.eval()
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)

    test_id = res['test_id']
    train_test_id = res['train_test_id']

    cls_seen_num = att_seen.shape[0]
    cls_unseen_num = att_unseen.shape[0]

    with torch.no_grad():
        acc_zsl, acc_gzsl_unseen, acc_gzsl_seen, H, AUSUC, best_gamma = eval(
            tu_loader,
            ts_loader,
            att_unseen,
            att_seen,
            cls_unseen_num,
            cls_seen_num,
            test_id,
            train_test_id,
            model,
            test_gamma,
            ReZSL,
            device
        )

    model.train()

    return acc_gzsl_seen, acc_gzsl_unseen, H, acc_zsl, AUSUC, best_gamma

def computePearson(semantics, error_matrix, dim):
    """
    :param semantics: [c,s]
    :param rezsl_error_matrix: [c,s]
    :return:
    """

    c, s = semantics.shape

    if dim == 1:
        semantics = semantics.T
        error_matrix = error_matrix.T
    pearson_size = semantics.shape[0]
    pearson = torch.zeros((pearson_size)).cuda()

    for i in range(pearson_size):
        semantic = semantics[i]
        semantic_mean = semantic.mean()
        error = error_matrix[i]
        error_mean = error.mean()
        numerator = ((semantic - semantic_mean.expand(semantic.shape)) * (error - error_mean.expand(error.shape))).sum()
        denominator = torch.sqrt((semantic - semantic_mean.expand(semantic.shape)).pow(2).sum()) * torch.sqrt(
            (error - error_mean.expand(error.shape)).pow(2).sum())
        r = numerator / (denominator + 1e-12)
        pearson[i] = r

    return pearson