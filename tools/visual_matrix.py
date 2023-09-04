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
from GEMZSL.config import cfg
from torch.utils.data import DataLoader, TensorDataset
from GEMZSL.modeling import AttentionNet, BasicNet, GEMNet, ReZSL, resnet101_features, get_attributes_info, ViT, \
    get_attr_group
from visual_tools import prepare_attri_label, ImageFilelist
import argparse
import pickle
import matplotlib.ticker as ticker
from GEMZSL.utils import ReDirectSTD, set_seed
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
opt.REZSL = True
opt.SEMANTIC_TYPE = "GBU"
opt.BACKBONE_TYPE = "vit"

if opt.dataset=="AWA2" and not (opt.BACKBONE_TYPE == "vit"):
    opt.hid_dim = 4096
else:
    opt.hid_dim = 0
opt.GPU = [0]


opt.cuda = True
opt.SEMANTIC = 'original'

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
        att_name_path = '/Data_HDD/ra_zihan_ye/datasets/CUB_200_2011/attributes.txt'
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
        att_name_path = '/Data_HDD/ra_zihan_ye/datasets/AWA2/Animals_with_Attributes2/predicates.txt'
        with open(att_name_path, 'r') as f:
            while True:
                record = f.readline()
                if len(record) == 0:
                    break
                att_name = record.split('	')[1][:-1]
                att_names.append(att_name)
    return att_names

def build_BasicNet(dataset_name, att_type, backbone_type):
    info = get_attributes_info(dataset_name, att_type)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]

    img_size = cfg.DATASETS.IMAGE_SIZE

    model_dir = "./pretrained_models"
    pretrained = True
    ft_flag = True
    if backbone_type == 'resnet':
        # res101 feature size
        img_size = 448
        c, w, h = 2048, img_size // 32, img_size // 32
        backbone = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    elif backbone_type == 'vit':
        # res101 feature size
        img_size = 224
        c, w, h = 768, img_size // 16, img_size // 16
        if img_size == 224:
            backbone = ViT(model_name="vit_base_patch16_224", pretrained=pretrained)
            # backbone = ViT(model_name="vit_large_patch16_224_in21k", pretrained=pretrained)
        else:  # img_size == 384
            backbone = ViT(model_name="vit_base_patch16_384", pretrained=pretrained)

    scale = 20.0

    device = torch.device("cuda")
    return BasicNet(backbone=backbone, backbone_type=backbone_type, ft_flag=ft_flag, img_size=img_size, hid_dim=opt.hid_dim,
                    c=c, w=w, h=h, scale=scale, attritube_num=attritube_num, cls_num=cls_num, ucls_num=ucls_num,
                    device=device)


def build_AttentionNet(dataset_name, att_type, backbone_type):
    info = get_attributes_info(dataset_name, att_type)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    attr_group = get_attr_group(dataset_name)
    img_size = 448
    # res101 feature size
    c, w, h = 2048, img_size // 32, img_size // 32

    scale = 20.0

    img_size = cfg.DATASETS.IMAGE_SIZE

    model_dir = "./pretrained_models"
    pretrained = True
    ft_flag = True
    if backbone_type == 'resnet':
        # res101 feature size
        img_size = 448
        c, w, h = 2048, img_size // 32, img_size // 32
        backbone = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    elif backbone_type == 'vit':
        # res101 feature size
        img_size = 224
        c, w, h = 768, img_size // 16, img_size // 16
        if img_size == 224:
            backbone = ViT(model_name="vit_base_patch16_224", pretrained=pretrained)
            # backbone = ViT(model_name="vit_large_patch16_224_in21k", pretrained=pretrained)
        else:  # img_size == 384
            backbone = ViT(model_name="vit_base_patch16_384", pretrained=pretrained)

    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device("cuda")

    return AttentionNet(backbone=backbone, backbone_type=backbone_type, ft_flag=ft_flag, img_size=img_size,
                        hid_dim=opt.hid_dim,
                        c=c, w=w, h=h, scale=scale,
                        attritube_num=attritube_num,
                        attr_group=attr_group, w2v=w2v,
                        cls_num=cls_num, ucls_num=ucls_num, device=device)


def build_GEMNet(dataset_name, att_type):
    dataset_name = dataset_name
    info = get_attributes_info(dataset_name, att_type)
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
    set_seed(214)
    device = torch.device("cuda")

    if opt.dataset == "AWA2":
        ymax = 0.05
        baseline_AttentionNet_path = "/Data_PHD/phd22_zihan_ye/project/ReZSL/GEM-ZSL-main/checkpoints/awa_16w_2s_original/AttentionNet(vit224,mid:300,hid:0)_SGD(lr=1e-4)_NCE+ReMSE(0.0,1.0)_Ablation_Global_seed=1514_test_epoch:29.pth"
    elif opt.dataset == "CUB":
        ymax = 0.05
        baseline_AttentionNet_path = "/Data_PHD/phd22_zihan_ye/project/ReZSL/GEM-ZSL-main/checkpoints/cub_16w_2s_original/AttentionNet(vit224,hid:0)_SGD(lr=5e-4)_NCE_Ablation_Global+Attentive+ReMSE(1.0,1.0)_seed=214.pth"
    elif opt.dataset == "SUN":
        ymax = 0.15
        baseline_AttentionNet_path = "/Data_PHD/phd22_zihan_ye/project/ReZSL/GEM-ZSL-main/checkpoints/sun_16w_2s_original/AttentionNet(vit224,hid:0)_SGD(lr=5e-4)_NCE_Ablation_Global+Attentive+0.01ReMSE(1.0,1.0)_seed=214.pth"


    attribute_names = read_attribute_name(opt.dataset)
    print(attribute_names)

    # model load
    model = build_AttentionNet(dataset_name=opt.dataset, att_type=opt.SEMANTIC_TYPE, backbone_type=opt.BACKBONE_TYPE)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=opt.GPU)
    model.load_state_dict(torch.load(baseline_AttentionNet_path)['model'])
    model.eval()
    print('successed load: '+baseline_AttentionNet_path)

    sample_attributes = data.attribute # cls_num, att_dim
    attribute_zsl = prepare_attri_label(sample_attributes, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(sample_attributes, data.seenclasses).cuda()
    attribute_all = sample_attributes.cuda()

    cls_num, att_dim = data.attribute.shape
    info = get_attributes_info(opt.dataset, att_type=opt.SEMANTIC_TYPE)
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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=4,pin_memory=True)

    print('evaluation started')
    matrix = test_gzsl(opt, model, test_loader, attribute_all, rezsl)
    print('evaluation finished')
    print(matrix)
    print(matrix.shape)
    print("mean:")
    print(matrix.mean())
    if opt.dataset =="CUB":
        vmin = 5.8364e-06
        vmax = 0.1104
    elif opt.dataset == "AWA2":
        vmin = 0.0001
        vmax = 0.2705
    elif opt.dataset == "SUN":
        vmin = 5.0641e-06
        vmax = 0.3323
    matrix, yticklabels = filter_nonzero(matrix)
    print("min: ")
    print(torch.min(matrix))
    print("max: ")
    print(torch.max(matrix))
    xylimits = {'xmin':230, 'xmax': 250,'ymin':0, 'ymax':20}
    vlimits = {'flag':True,'vmin':vmin,'vmax':vmax}
    #draw_matrix(matrix.cpu().numpy(), vlimits = vlimits,log_flag=False)
    draw_matrix(matrix.cpu().numpy(), vlimits = vlimits, log_flag=True)
    #for i in range(cls_num):
        #draw_attribute_error(matrix, attribute_names, dataset=opt.dataset, class_id=i, flag=False, use_sort=False, use_log=False)
    print('draw finished')

def filter_nonzero(matrix):
    cls_num,att_num = matrix.shape
    yticklabels = []
    count = 0
    for i in range(cls_num):
        if matrix[i][0] > 0.0:
            count = count+1
    filter_nonzero_matrix = torch.zeros(count, att_num, requires_grad=False).to(matrix.device)
    count = 0
    for i in range(cls_num):
        if matrix[i][0] > 0.0:
            filter_nonzero_matrix[count] = matrix[i]
            yticklabels.append(str(i))
            count = count+1
    return filter_nonzero_matrix, yticklabels


def test_gzsl(opt, model, testloader, attribute, rezsl_weights):
    rezsl_weights.afterTest()
    with torch.no_grad():
        num = len(testloader)
        for i, (batch_input, batch_labels, batch_impath) in \
                enumerate(testloader):
            print('%.d/%.d'%(i,num))
            target_attri = attribute[batch_labels, :]
            if opt.cuda:
                batch_input = batch_input.cuda()
                batch_labels = batch_labels.cuda()
                target_attri = target_attri.cuda()
            v2s = model(x=batch_input, support_att=attribute)
            rezsl_weights.arrangeTestOffset(v2s.detach(), target_attri.detach(), batch_labels.detach())
    rezsl_weights.averageTestOffset()
    return rezsl_weights.test_cls_offset_mean

def draw_matrix(matrix, vlimits, log_flag=False):
    plt.cla()
    plt.rcParams['savefig.dpi'] = 600
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #cax = ax.matshow(matrix, cmap='coolwarm')
    if log_flag==True:
        matrix = np.log(matrix)
        vlimits['vmin'], vlimits['vmax'] = np.log(vlimits['vmin']), np.log(vlimits['vmax'])

    if vlimits['flag']==True:
        cax = ax.matshow(matrix, cmap='coolwarm', vmin=vlimits['vmin'], vmax=vlimits['vmax'])
    else:
        cax = ax.matshow(matrix, cmap='coolwarm')

    clb = fig.colorbar(cax,shrink=0.72)
    clb.ax.tick_params(labelsize=6)
    plt.tick_params(labelsize=4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    #ax.set_yticks(range(0,len(yticklabels), 1))
    #print(yticklabels)
    #ax.set_yticklabels(yticklabels)

    plt.xlabel("Semantic", fontsize=8)
    plt.ylabel("Test class", fontsize=8)
    if log_flag ==True:
        if opt.REG_NORM:
            if opt.REZSL:
                fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_Re" +opt.REG_TYPE+"_"+opt.BACKBONE_TYPE + "_log.png"
            else:
                fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_N" + opt.REG_TYPE +"_"+opt.BACKBONE_TYPE + "_log.png"
        else:
            fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_" + opt.REG_TYPE +"_"+opt.BACKBONE_TYPE + "_log.png"
    else:
        if opt.REG_NORM:
            if opt.REZSL:
                fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_Re" + opt.REG_TYPE +"_"+opt.BACKBONE_TYPE + ".png"
            else:
                fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_N" + opt.REG_TYPE +"_"+opt.BACKBONE_TYPE + ".png"
        else:
            fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_" + opt.REG_TYPE +"_"+opt.BACKBONE_TYPE + ".png"
    plt.savefig(fig_path, bbox_inches='tight')
    print('saved in '+fig_path)

def draw_AWA2_matrix(matrix,yticklabels,vmin,vmax):
    plt.cla()
    plt.rcParams['savefig.dpi'] = 600
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #cax = ax.matshow(matrix, cmap='coolwarm')
    #cax = ax.matshow(matrix, cmap='coolwarm',vmin=vmin, vmax=vmax)
    matrix = np.log(matrix)
    limits = False
    if limits == True:
        cax = ax.matshow(matrix, cmap='coolwarm', vmin=np.log(vmin), vmax=np.log(vmax))
    else:
        cax = ax.matshow(matrix, cmap='coolwarm')
    clb = fig.colorbar(cax,shrink=0.72)
    clb.ax.tick_params(labelsize=6)
    plt.tick_params(labelsize=4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticks(range(0,len(yticklabels), 1))
    print(yticklabels)
    ax.set_yticklabels(yticklabels)

    plt.xlabel("Semantic", fontsize=10)
    plt.ylabel("Test class", fontsize=10)
    if opt.REG_NORM == True:
        fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_N" + opt.REG_TYPE+"_log.png"
    else:
        fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_" + opt.REG_TYPE + "_log.png"
    plt.savefig(fig_path, bbox_inches='tight')
    print('saved in '+fig_path)

def draw_partial_matrix(matrix, xylimits, vlimits):
    plt.cla()
    plt.rcParams['savefig.dpi'] = 600
    fig = plt.figure()
    ax = fig.add_subplot(111)

    matrix = np.log(matrix)
    vlimits['vmin'], vlimits['vmax'] = np.log(vlimits['vmin']), np.log(vlimits['vmax'])

    if vlimits['flag']==True:
        cax = ax.matshow(matrix, cmap='coolwarm', vmin=vlimits['vmin'], vmax=vlimits['vmax'])
    else:
        cax = ax.matshow(matrix, cmap='coolwarm')

    clb = fig.colorbar(cax,shrink=1.0)
    clb.ax.tick_params(labelsize=6)
    plt.tick_params(labelsize=6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(xylimits['xmin'], xylimits['xmax'])
    ax.set_ylim(xylimits['ymin'], xylimits['ymax'])

    plt.xlabel("Semantic", fontsize=12)
    plt.ylabel("Test class", fontsize=12)
    if opt.REG_NORM == True:
        fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_N" +opt.REG_TYPE+"_x("+str(xylimits['xmin'])+","+str(xylimits['xmax'])+")"+"_y("+str(xylimits['ymin'])+","+str(xylimits['ymax'])+")"+"_log.png"
    else:
        fig_path = "./Statistics/matrix/" + opt.dataset + "/" + "matrix_" + opt.REG_TYPE + "_x(" + str(xylimits['xmin']) + "," + str(xylimits['xmax']) + ")" + "_y(" + str(xylimits['ymin']) + "," + str(xylimits['ymax']) + ")" + "_log.png"
    plt.savefig(fig_path, bbox_inches='tight')
    print('saved in '+fig_path)

def draw_attribute_error(matrix, attribute_names, dataset, class_id, flag=False, use_sort=False, use_log=False):

    plt.cla()
    plt.rcParams['savefig.dpi'] = 600
    fig = plt.figure(figsize=(12,6)) #width, height
    ax = fig.add_subplot(111)

    #attribute_errors = matrix.mean(dim=0)
    if use_log:
        attribute_errors = -torch.log(matrix[class_id])
    else:
        attribute_errors = matrix[class_id]
    names_recorder = []
    errors_recorder = []
    if use_sort ==True:
        errors_recorder, indices = attribute_errors.sort(0, descending=True)
        errors_recorder = errors_recorder.cpu().numpy()

        for i in indices:
            names_recorder.append(attribute_names[indices[i]])
    else:
        errors_recorder = attribute_errors.cpu().numpy()
        for attribute_name in attribute_names:
            names_recorder.append(attribute_name)

    plt.bar(names_recorder, errors_recorder, width=0.6)
    if dataset=="CUB":
        plt.xticks(rotation=60,fontsize = 2) # fontsize: CUB,2
        if flag == True:
            ax.set_ylim(0, 0.030)
    else:
        plt.xticks(rotation=60, fontsize=4)

    if opt.REG_NORM == True:
        if opt.REZSL:
            fig_path = "./Statistics/AttributeErrorBar/" + opt.dataset + "/" + "AttributeErrorBar_Re" + opt.REG_TYPE + "_class"+str(class_id)+".png"
        else:
            fig_path = "./Statistics/AttributeErrorBar/" + opt.dataset + "/" + "AttributeErrorBar_N" + opt.REG_TYPE + "_class"+str(class_id)+".png"
    else:
        fig_path = "./Statistics/AttributeErrorBar/" + opt.dataset + "/" + "AttributeErrorBar_" + opt.REG_TYPE + "_class"+str(class_id)+".png"
    plt.savefig(fig_path, bbox_inches='tight')
    print('saved in ' + fig_path)

main()