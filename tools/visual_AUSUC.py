# -*- coding: utf-8 -*-
import os
from os.path import join
import sys
import argparse

sys.path.append("/ReZSL-main")
import scipy.io as sio
import random
import time
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import argparse
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from REZSL.data import DATA_LOADER,build_dataloader
import torchvision.transforms as transforms
from REZSL.config import cfg
from REZSL.modeling import AttentionNet, BasicNet, GEMNet, ReZSL, resnet101_features, get_attributes_info, get_attr_group
import argparse
import pickle
import matplotlib.ticker as ticker
import pandas as pd
from pandas import DataFrame
from visual_tools import prepare_attri_label, ImageFilelist
from REZSL.config import cfg
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(color_codes=True)

#fontsize = 16
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
#plt.rcParams['font.size'] = fontsize
#plt.xticks(fontsize=fontsize)
#plt.yticks(fontsize=fontsize)

cudnn.benchmark = True

parser = argparse.ArgumentParser()
opt = parser.parse_args()
opt.matdataset = True
opt.gzsl = True
opt.preprocessing = True
opt.validation = False
opt.standardization = False
opt.dataroot = '/Data_HDD/ra_zihan_ye/datasets/xlsa17/data'
opt.dataset = 'SUN'  # CUB AWA2 SUN
opt.method = 'AttentionNet' # GEM BasicNet AttentionNet
opt.image_embedding = 'res101'
opt.class_embedding = 'att'
opt.image_root = '/Data_HDD/ra_zihan_ye/datasets/'
opt.batch_size = 32
opt.REG_NORM = True
opt.REG_TYPE = "MSE"
opt.REZSL = False
opt.GPU = [0]
opt.SEMANTIC = "normalized"
if opt.dataset=='AWA2':
    opt.hid_dim = 4096
else:
    opt.hid_dim = 0

opt.cuda = True

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1

parser = argparse.ArgumentParser(description="PyTorch Zero-Shot Learning Training")
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

data = DATA_LOADER(opt)
seenclasses = data.seenclasses
unseenclasses = data.unseenclasses
if opt.gzsl:
    testclsnum = data.allclasses.size(0)
else:
    testclsnum = data.unseenclasses.size(0)

attribute = data.attribute
att_dim = data.att_dim
vis_dim = data.feature_dim

attribute_shape = attribute.shape
train_visual = data.train_feature
seenclasses_attribute = data.train_att
all_visual = torch.cat((data.train_feature, data.test_seen_feature, data.test_unseen_feature), dim=0)
if opt.gzsl:
    test_visual = torch.cat((data.test_seen_feature, data.test_unseen_feature), dim=0)
    test_label = torch.cat((data.test_seen_label, data.test_unseen_label), dim=0)
else:
    test_visual = data.test_unseen_feature
    test_label = data.test_unseen_label
print(test_visual.shape)
print(test_label.shape)

def build_BasicNet(dataset_name):
    dataset_name = dataset_name
    info = get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    img_size = cfg.DATASETS.IMAGE_SIZE
    attr_group = get_attr_group(dataset_name)
    img_size = 448
    # res101 feature size
    c, w, h = 2048, img_size // 32, img_size // 32

    # res101 feature size
    scale = 20.0

    pretrained = True
    model_dir = "./pretrained_models"
    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    ft_flag = True

    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device("cuda")

    return BasicNet(res101=res101, ft_flag=ft_flag, img_size=img_size, hid_dim=opt.hid_dim,
                    c=c, w=w, h=h, scale=scale,
                    attritube_num=attritube_num,
                    attr_group=attr_group, w2v=w2v,
                    cls_num=cls_num, ucls_num=ucls_num, device=device)


def build_AttentionNet(dataset_name):
    dataset_name = dataset_name
    info = get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    attr_group = get_attr_group(dataset_name)
    img_size = 448
    # res101 feature size
    c, w, h = 2048, img_size // 32, img_size // 32

    scale = 20.0

    pretrained = True
    model_dir = "./pretrained_models"
    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    ft_flag = True

    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device("cuda")

    return AttentionNet(res101=res101, ft_flag=ft_flag, img_size=img_size, hid_dim=opt.hid_dim,
                        c=c, w=w, h=h, scale=scale,
                        attritube_num=attritube_num,
                        attr_group=attr_group, w2v=w2v,
                        cls_num=cls_num, ucls_num=ucls_num, device=device)


def build_GEMNet(dataset_name):
    dataset_name = dataset_name
    info = get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    attr_group = get_attr_group(dataset_name)
    img_size = 448
    # res101 feature size
    c, w, h = 2048, img_size // 32, img_size // 32

    scale = 20.0

    pretrained = True
    model_dir = "./pretrained_models"
    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    ft_flag = True

    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device("cuda")

    return GEMNet(res101=res101, ft_flag=ft_flag, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num, device=device)


def main():
    device = torch.device("cuda")

    cls_num, att_dim = data.attribute.shape
    info = get_attributes_info(opt.dataset)
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num
    rezsl = ReZSL(p=0.0, p2=0.0, momentum=0.0, att_dim=att_dim, train_class_num=scls_num, test_class_num=cls_num,
                  RegNorm=opt.REG_NORM, RegType=opt.REG_TYPE, device=device)

    opt.test_seen_label = data.test_seen_label
    print('ImageFilelist')

    if opt.dataset == "AWA2":
        attentionNet_model_path = './checkpoints/awa_16w_2s_original/AttentionNet(mid:300,hid:4096)_SGD(lr=1e-4)_NCE+0.1NMSE_seed=1514_AUSUC.pth'  # AttentionNet
        attentionNet_remse_model_path = './checkpoints/awa_16w_2s_original/AttentionNet(mid:300,hid:4096)_SGD(lr=1e-4)_NCE+0.1ReMSE(0.0,0.25)_seed=1514.pth'  # AttentionNet+ReMSE
        cfg.DATASETS.NAME = "AWA2"
    elif opt.dataset == "CUB":
        attentionNet_model_path = './checkpoints/cub_16w_2s_original/AttentionNet(hid:0)_SGD(lr=5e-4)_NCE+ReMSE(0.0,0.0)_seed=214_AUSUC.pth' # AttentionNet
        attentionNet_remse_model_path = './checkpoints/cub_16w_2s_original/AttentionNet(hid:0)_SGD(lr=5e-4)_NCE+ReMSE(2.0,0.0)_seed=214_AUSUC.pth'  # AttentionNet+ReMSE
        cfg.DATASETS.NAME = "CUB"
    cfg.DATASETS.SEMANTIC = "normalized"
    cfg.DATASETS.WAYS = 16
    cfg.DATASETS.SHOTS = 2
    cfg.SOLVER.DATA_AUG = "resize_random_crop"
    cfg.DATASETS.IMAGE_SIZE = 448
    cfg.TEST.IMS_PER_BATCH = 32
    cfg.MODEL.LOSS.REG_NORM = True
    cfg.MODEL.LOSS.REG_TYPE = "MSE"
    print(cfg.DATASETS.NAME)
    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(cfg, is_distributed=distributed)
    train_test_id = res['train_test_id']
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)
    att = torch.cat((att_seen, att_unseen), dim=0)

    if opt.method=="AttentionNet":
        model = build_AttentionNet(dataset_name=opt.dataset)
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=opt.GPU)
        model_path = attentionNet_model_path
        # bmc_model_path = attentionNet_bmc_model_path
        remse_model_path = attentionNet_remse_model_path
    elif opt.method=="GEM":
        model = build_GEMNet(dataset_name=opt.dataset)
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=opt.GPU)
        model_path = gem_model_path
        # bmc_model_path = gem_bmc_model_path
        remse_model_path = gem_remse_model_path

    # model load
    print('Vanilla model')
    AUSUC, AUSUC_vector, start_tu = loadAndAUSUC(model, model_path, ts_loader, tu_loader, att,scls_num, ucls_num, train_test_id, rezsl)
    print('AUSUC')
    print(AUSUC)
    # bmc model load
    #bmc_AUSUC_vector, bmc_start_tu = loadAndAUSUC(model, bmc_model_path, ts_loader, tu_loader, att, scls_num, ucls_num, train_test_id, rezsl)

    # remse model load
    print('ReMSE model')
    remse_AUSUC, remse_AUSUC_vector, remse_start_tu = loadAndAUSUC(model, remse_model_path, ts_loader, tu_loader, att, scls_num, ucls_num, train_test_id, rezsl)
    print('remse AUSUC')
    print(remse_AUSUC.mean())
    draw_AUSUC_plot(AUSUC_vector, start_tu, '#82B0D2', r'GEMZSL')
    #draw_AUSUC_plot(gem_bmc_AUSUC_vector, gem_bmc_start_tu, '#82B0D2')
    draw_AUSUC_plot(remse_AUSUC_vector, remse_start_tu, '#FA7F6F', r'GEMZSL+ReMSE')
    plt.xlim(xmin=0.0)
    plt.ylim(ymin=0.0)
    plt.xlabel(r'Unseen accuracy (%)')
    plt.ylabel(r'Seen accuracy (%)')
    plt.legend()
    plt.savefig('./Statistics/AUSUC/' + opt.dataset + '/AUSUC.png', bbox_inches='tight')

def loadAndAUSUC(model,model_path,ts_loader, tu_loader, att, scls_num, ucls_num, train_test_id, rezsl):
    model.load_state_dict(torch.load(model_path)['model'], strict=False)
    model.eval()
    error_matrix, AUSUC, AUSUC_vector, start_H, start_ts, start_tu, best_H, best_ts, best_tu, _ = cal_AUSUC(model, ts_loader, tu_loader, att, scls_num, ucls_num, train_test_id, rezsl, device='cuda')
    rezsl.afterTest()
    return AUSUC, AUSUC_vector, start_tu

def draw_AUSUC_plot(AUSUC_vector,start_tu,color,label):
    '''
    :param AUSUC_vector: [n,2] # seen, unseen
    :param start_H,start_ts,start_tu_acc: float
    :return:
    '''
    unseen = []
    seen = []
    anchor = 0
    AUSUC_vector_sorted = sorted(AUSUC_vector,key =lambda x:x[0])
    for i in range(len(AUSUC_vector_sorted)):
        unseen.append(AUSUC_vector_sorted[i][1]*100)
        seen.append(AUSUC_vector_sorted[i][0]*100)
        if AUSUC_vector_sorted[i][1] == start_tu:
            AUSUC_start_x = AUSUC_vector_sorted[i][1]*100
            AUSUC_start_y = AUSUC_vector_sorted[i][0]*100
    plt.plot(unseen, seen, label=label, color=color)
    plt.scatter(AUSUC_start_x, AUSUC_start_y, label = label+" Start Point", c=color, marker='x', s=100)


def test_optimize_score(build_function, model_path, test_loader, attribute_all, rezsl, device):
    model = build_function(dataset_name=opt.dataset)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=opt.GPU)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    print('successed load')
    print('evaluation started')
    rezsl = test_gzsl(opt, model, test_loader, attribute_all, rezsl)
    print('evaluation finished')
    error_matrix = rezsl.test_cls_offset_mean
    return error_matrix


def test_gzsl(opt, model, testloader, attribute, rezsl):
    rezsl.afterTest()
    step_num = len(testloader)
    with torch.no_grad():
        for i, (batch_input, batch_labels, batch_impath) in \
                enumerate(testloader):
            print("%.d/%.d"%(i,step_num))
            target_attri = attribute[batch_labels, :]
            if opt.cuda:
                batch_input = batch_input.cuda()
                batch_labels = batch_labels.cuda()
                target_attri = target_attri.cuda()
            pre_attri = model(x=batch_input, support_att=attribute)
            rezsl.arrangeTestOffset(pre_attri.detach(), target_attri.detach(), batch_labels.detach())

    rezsl.averageTestOffset()

    return rezsl

def stack_cal(scores,bias):
    if not scores.shape[1] == bias.shape[1]:
        bias = bias.repeat(1, len(range(torch.cuda.device_count())))
    scores = scores - bias
    return scores

def inference(model,dataloadr,support_att,ReZSL,device):
    scores = []
    labels = []
    step_num = len(dataloadr)
    with torch.no_grad():
        for i, (img, label_att, label) in enumerate(dataloadr):
            img = img.to(device)
            label = label.to(device)
            label_att = support_att[label, :]
            label_att = label_att.to(device)
            print("%.d/%.d"%(i,step_num))
            v2s = model(x=img, support_att=support_att, )
            try:
                score, cos_dist = model.module.cosine_dis(pred_att=v2s, support_att=support_att)
            except Exception as E:
                score, cos_dist = model.cosine_dis(pred_att=v2s, support_att=support_att)

            _, pred = score.max(dim=1)
            scores.append(score)
            labels.append(label)
            ReZSL.arrangeTestOffset(v2s.detach(), label_att.detach(), label.detach())

    TestOffsetMatrix, Test_mean_value = ReZSL.averageTestOffset()
    # print(str(Test_mean_value) + ' overall: ' + str(torch.mean(Test_mean_value)))
    mask = torch.gt(Test_mean_value, 0.0)
    mean = torch.mean(torch.masked_select(Test_mean_value, mask))
    std = torch.std(torch.masked_select(Test_mean_value, mask))
    print('Test_mean_offset mean: ' + str(mean.item())+'. std: ' + str(std.item())+'.')
    ReZSL.afterTest()

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
        test_gamma_interval = -0.1

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
        test_gamma_interval = 0.1

        while True:
            test_gamma = test_gamma + test_gamma_interval
            bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma)
            bias_u = torch.zeros((1, cls_unseen_num))
            bias = torch.cat([bias_s, bias_u], dim=1).to(device)
            ts_scores, tu_scores = stack_cal(start_ts_scores, bias), stack_cal(start_tu_scores, bias)
            ts_acc, tu_acc = get_accuracy(ts_scores, ts_labels, test_id), get_accuracy(tu_scores, tu_labels, test_id)
            AUSUC_vector.append((ts_acc, tu_acc)) # seen,unseen
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

    return ReZSL.test_cls_offset_mean, AUSUC, AUSUC_vector, start_H, start_ts_acc, start_tu_acc, best_H, best_ts, best_tu, best_gamma

main()