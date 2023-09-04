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
from torch.utils.data import DataLoader,TensorDataset
import argparse
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from GEMZSL.data import DATA_LOADER
import torchvision.transforms as transforms
from REZSL.config import cfg
from REZSL.modeling import AttentionNet, BasicNet, GEMNet, ReZSL, resnet101_features, get_attributes_info, get_attr_group
import argparse
import pickle
import matplotlib.ticker as ticker
from visual_tools import prepare_attri_label, ImageFilelist
cudnn.benchmark = True

parser = argparse.ArgumentParser()
opt = parser.parse_args()
opt.matdataset = True
opt.gzsl = True
opt.preprocessing = True
opt.validation = False
opt.standardization = False
opt.dataroot = '/Datasets/xlsa17/data'
opt.dataset = 'AWA2' # CUB AWA2 SUN
opt.image_embedding = 'res101'
opt.class_embedding = 'att'
opt.image_root = '/Datasets/'
opt.batch_size = 32
opt.REG_NORM = True
opt.REG_TYPE = "MSE"
opt.REZSL = False
opt.GPU = [0]
opt.SEMANTIC = "normalized"
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
all_visual = torch.cat((data.train_feature,data.test_seen_feature,data.test_unseen_feature),dim=0)
if opt.gzsl:
    test_visual = torch.cat((data.test_seen_feature,data.test_unseen_feature),dim=0)
    test_label = torch.cat((data.test_seen_label,data.test_unseen_label),dim=0)
else:
    test_visual = data.test_unseen_feature
    test_label = data.test_unseen_label
print(test_visual.shape)
print(test_label.shape)

opt.positive = False

def read_attribute_name(dataset):
    att_names = []
    if dataset == 'CUB':
        att_name_path = '/Datasets/CUB_200_2011/attributes.txt'
        with open(att_name_path,'r') as f:
            while True:
                record = f.readline()
                if len(record) == 0:
                    break
                att_name = record.split(' ')[1][:-1]
                att_name = att_name.replace('has_','')
                att_name = att_name.replace('_color', '')
                att_name = att_name.replace('_shape', '')
                att_name = att_name.replace('_pattern', '')
                att_names.append(att_name)
    elif dataset == 'AWA2':
        att_name_path = '/Datasets/AWA2/Animals_with_Attributes2/predicates.txt'
        with open(att_name_path, 'r') as f:
            while True:
                record = f.readline()
                if len(record) == 0:
                    break
                att_name = record.split('	')[1][:-1]
                att_names.append(att_name)
    return att_names

def build_BasicNet(dataset_name):
    dataset_name = dataset_name
    info = get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num-ucls_num

    attr_group = get_attr_group(dataset_name)

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

    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device("cuda")

    return BasicNet(res101=res101, ft_flag = ft_flag, img_size=img_size, hid_dim=opt.hid_dim,
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
    scls_num = cls_num-ucls_num

    attr_group = utils.get_attr_group(dataset_name)
    img_size = 448
    # res101 feature size
    c,w,h = 2048, img_size//32, img_size//32

    scale = 20.0

    pretrained = True
    model_dir = "./pretrained_models"
    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    ft_flag = True

    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device("cuda")

    return AttentionNet(res101=res101, ft_flag = ft_flag, img_size=img_size, hid_dim=opt.hid_dim,
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
    scls_num = cls_num-ucls_num

    attr_group = get_attr_group(dataset_name)
    img_size = 448
    # res101 feature size
    c,w,h = 2048, img_size//32, img_size//32

    scale = 20.0

    pretrained = True
    model_dir = "./pretrained_models"
    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    ft_flag = True

    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device("cuda")

    return GEMNet(res101=res101, ft_flag = ft_flag, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num, device=device)

def main():
    device = torch.device("cuda")
    attribute_names = read_attribute_name(opt.dataset)
    print(attribute_names)

    test_batch = 32

    sample_attributes = data.attribute # cls_num, att_dim
    attribute_zsl = prepare_attri_label(sample_attributes, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(sample_attributes, data.seenclasses).cuda()
    attribute_all = sample_attributes.cuda()

    cls_num, att_dim = data.attribute.shape
    info = get_attributes_info(opt.dataset)
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num
    rezsl = ReZSL(p=0.0, p2=0.0, momentum=0.0, att_dim=att_dim, train_class_num=scls_num, test_class_num=cls_num, RegNorm=opt.REG_NORM, RegType=opt.REG_TYPE, device=device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize, ])
    opt.test_seen_label = data.test_seen_label
    print('ImageFilelist')
    if opt.gzsl:
        test_data = ImageFilelist(opt, data_inf=data,
                                      transform=test_transform,
                                      dataset=opt.dataset,
                                      image_type='test_loc')
    else:
        test_data = ImageFilelist(opt, data_inf=data,
                                  transform=test_transform,
                                  dataset=opt.dataset,
                                  image_type='test_unseen_loc')

    if opt.dataset=="AWA2":
        gem_model_path = './checkpoints/awa_16w_2s_original/GEMNet_SGD(lr=1e-3)_NCE+ReMSE(0.5,0.05)_seed=214.pth'  # GEMZSL+ReMSE
    elif opt.dataset=="CUB":
        gem_model_path = './checkpoints/cub_16w_2s_original/GEMNet_SGD(lr=1e-3)_NCE+MSE+0.1AD+0.2CPT_seed=214.pth' # GEMZSL
    elif opt.dataset == "SUN":
        gem_model_path = './checkpoints/sun_16w_2s_original/GEMNet_SGD(lr=1e-3)_NCE+BMC_seed=214.pth'  # GEMZSL+BMC

    # rezsl model load
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=4,pin_memory=True)
    gem_matrix, gem_max_value, gem_max_indice, gem_incorrect_MSE, gem_incorrect2gt_cos, gem_incorrect2mis_cos, gem_incorrect_count, gem_cos_gap, gem_gt2mis_cos, gem_gt2close_cos, gem_incorrect2close_count = test_optimize_score(build_GEMNet, gem_model_path, test_loader, attribute_all, rezsl, device)
    rezsl.afterTest()

    print('rezsl')
    #for i in range(gem_max_indice.shape[0]):
        #print(attribute_names[i] + ": " + str(gem_max_value[i].data))
    gem_pearson = computePearson(attribute_all, gem_matrix, dim=0)
    print("pearson: %.8f"%(gem_pearson.mean()))
    print("MSE: %.8f"%(gem_matrix.mean()))



def filter_out(error_matrix, semantics, test_cls_count):
    n, s = error_matrix.shape
    filtered_error_matrix = torch.zeros((0,s)).to(error_matrix.device)
    filtered_semantics = torch.zeros((0,s)).to(semantics.device)
    print(test_cls_count)
    for i in range(n):
        if not test_cls_count[i] ==0:
            filtered_error_matrix = torch.cat([filtered_error_matrix, error_matrix[i].unsqueeze(0)],dim=0)
            filtered_semantics = torch.cat([filtered_semantics,semantics[i].unsqueeze(0)],dim=0)
        else:
            print("filter out class: %.d"%(i))
    return filtered_error_matrix, filtered_semantics



def test_optimize_rate(model_path, test_loader, attribute_all, rezsl, device):
    print("loading " + model_path)
    error_matrix_pre, _, _ = test_optimize_score(model_path, test_loader, attribute_all, rezsl, device)
    rezsl.afterTest()
    error_rate_pearson_vector = []
    for i in range(1, 30):
        model_path = model_path.split("_epoch_")[0] + "_epoch_" + str(i) + ".pth"
        print("loading " + model_path)
        error_matrix_current, _, _ = test_optimize_score(model_path, test_loader, attribute_all, rezsl, device)
        rate_matrix = error_matrix_current / (error_matrix_pre + 1e-12)
        rate_pearson = computePearson(attribute_all, rate_matrix, dim=0)
        error_matrix_pre = error_matrix_current
        print(rate_pearson.mean())  # more close to 1 indicate a stronger positive relation between the size of semantic and error rate, i.e. size~(cur/pre)
        error_rate_pearson_vector.append(rate_pearson.mean())
        rezsl.afterTest()
    return error_rate_pearson_vector


def test_optimize_score(build_function, model_path, test_loader, attribute_all, rezsl, device):
    model = build_function(dataset_name=opt.dataset)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=opt.GPU)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    print('successed load')
    print('evaluation started')
    incorrect_MSE, incorrect2gt_cos,incorrect2mis_cos, incorrect_count, cos_gap, gt2mis_cos, gt2close_cos, incorrect2close_count = test_gzsl(opt, model, test_loader, attribute_all, rezsl)
    print('evaluation finished')
    error_matrix = rezsl.test_cls_offset_mean
    error_matrix_mean = error_matrix.mean(dim=0)
    error_max_value, error_max_indice = error_matrix_mean.topk(k=20, dim=0, largest=True)
    return error_matrix, error_max_value, error_max_indice, incorrect_MSE, incorrect2gt_cos,incorrect2mis_cos, incorrect_count, cos_gap, gt2mis_cos, gt2close_cos, incorrect2close_count

def test_gzsl(opt, model, testloader, attribute, rezsl):
    rezsl.afterTest()

    incorrect_MSE = 0.0
    incorrect2gt_cos = 0.0
    incorrect2mis_cos = 0.0
    gt2mis_cos = 0.0
    gt2close_cos = 0.0
    incorrect_count = 0
    cos_gap = 0.0
    incorrect2close_count = 0

    with torch.no_grad():
        for i, (batch_input, batch_labels, batch_impath) in \
                enumerate(testloader):
            print(i)
            target_attri = attribute[batch_labels, :]

            if opt.cuda:
                batch_input = batch_input.cuda()
                batch_labels = batch_labels.cuda()
                target_attri = target_attri.cuda()
            pre_attri = model(x=batch_input, support_att=attribute)
            rezsl.arrangeTestOffset(pre_attri.detach(), target_attri.detach(), batch_labels.detach())

            n, s = pre_attri.shape
            if model.module == None:
                score, cos = model.cosine_dis(pred_att=pre_attri, support_att=attribute)
            else:
                score, cos = model.module.cosine_dis(pred_att=pre_attri, support_att=attribute)
            _, pred = score.max(dim=1)
            for j in range(n):
                if not pred[j] == batch_labels[j]:
                    incorrect_MSE = incorrect_MSE + torch.mean((pre_attri - target_attri[j]) ** 2)
                    incorrect2gt_cos = incorrect2gt_cos + cos[j][batch_labels[j]]
                    incorrect2mis_cos = incorrect2mis_cos + cos[j][pred[j]]
                    incorrect_count = incorrect_count+1
                    print("classified incorrectly")
                    print("cos pred-gt")
                    print(cos[j][batch_labels[j]])
                    print("cos pred-mis")
                    print(cos[j][pred[j]])
                    cos_gap = cos_gap + torch.abs(cos[j][batch_labels[j]] - cos[j][pred[j]])

                    gt = attribute[batch_labels[j]]
                    mis = attribute[pred[j]]
                    gt_norm = torch.norm(gt, p=2).expand_as(gt)
                    gt_normalized = gt.div(gt_norm + 1e-10)
                    mis_norm = torch.norm(mis, p=2).expand_as(mis)
                    mis_normalized = mis.div(mis_norm + 1e-10)
                    gt2mis_cos_dist = (gt_normalized*mis_normalized).sum()
                    gt2mis_cos = gt2mis_cos + gt2mis_cos_dist
                    print("cos gt-mis")
                    print(gt2mis_cos_dist)

                    gt_normalized_expand = gt.expand_as(attribute) # [c,s]
                    attribute_norm = attribute.norm(p=2, dim=1, keepdim=True).expand_as(attribute) # [c,s]
                    attribute_normalized = attribute / (attribute_norm + 1e-10)
                    top2_cos,_ = (gt_normalized_expand *attribute_normalized).sum(dim=1).topk(k=2, largest=True)
                    print("cos gt-close")
                    print(top2_cos[1])
                    gt2close_cos = gt2close_cos + top2_cos[1]

                    if top2_cos[1] == gt2mis_cos_dist:
                        incorrect2close_count = incorrect2close_count+1


    rezsl.averageTestOffset()

    return incorrect_MSE, incorrect2gt_cos,incorrect2mis_cos, incorrect_count, cos_gap, gt2mis_cos, gt2close_cos, incorrect2close_count

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
        numerator = ((semantic-semantic_mean.expand(semantic.shape)) * (error-error_mean.expand(error.shape))).sum()
        denominator = torch.sqrt( (semantic-semantic_mean.expand(semantic.shape)).pow(2).sum() ) * torch.sqrt( (error-error_mean.expand(error.shape)).pow(2).sum() )
        r = numerator / (denominator + 1e-12)
        pearson[i] = r

    return pearson


main()