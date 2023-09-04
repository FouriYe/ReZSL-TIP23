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
from REZSL.data import DATA_LOADER
import torchvision.transforms as transforms
from REZSL.config import cfg
from REZSL.modeling import AttentionNet, BasicNet, GEMNet, ReZSL, resnet101_features, get_attributes_info, get_attr_group
import argparse
import pickle
import matplotlib.ticker as ticker
from visual_tools import prepare_attri_label, ImageFilelist, default_flist_reader
from sklearn.linear_model import LinearRegression
import seaborn as sns
import cv2
from PIL import Image

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600

cudnn.benchmark = True

parser = argparse.ArgumentParser()
opt = parser.parse_args()
opt.matdataset = True
opt.gzsl = True
opt.preprocessing = True
opt.validation = False
opt.standardization = False
opt.dataroot = '/Datasets/xlsa17/data'
opt.dataset = 'AWA2'  # CUB AWA2 SUN
opt.method = 'AttentionNet' # GEM, BasicNet, AttentionNet
opt.image_embedding = 'res101'
opt.class_embedding = 'att'
opt.image_root = '/Datasets/'
opt.batch_size = 32
opt.REG_NORM = True
opt.REG_TYPE = "MSE"
opt.REZSL = False
opt.GPU = [0]
opt.SEMANTIC = "original"
opt.hid_dim = 0
if opt.dataset == 'AWA2':
    opt.hid_dim = 4096

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


def main():
    device = torch.device("cuda")

    test_batch = 32

    attribute_names = read_attribute_name(opt.dataset)
    print(attribute_names)
    class_names = data.class_names.tolist()

    sample_attributes = data.attribute  # cls_num, att_dim
    attribute_zsl = prepare_attri_label(sample_attributes, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(sample_attributes, data.seenclasses).cuda()
    attribute_all = sample_attributes.cuda()
    print(attribute_all.view(-1).shape)

    cls_num, att_dim = data.attribute.shape
    info = get_attributes_info(opt.dataset)
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num
    opt.test_seen_label = data.test_seen_label

    if opt.dataset == "AWA2":
        attention_model_path = './checkpoints/awa_16w_2s_original/AttentionNet(mid:300,hid:4096)_SGD(lr=1e-4)_NCE+0.1ReMSE(2.0,0.0)_seed=1514.pth'
    elif opt.dataset == "CUB":
        attention_model_path = './checkpoints/cub_16w_2s_original/AttentionNet(hid:0)_SGD(lr=5e-4)_NCE+NMSE_seed=214.pth'
    elif opt.dataset == "SUN":
        attention_model_path = './checkpoints/sun_16w_2s_original/AttentionNet(hid:0)_SGD(lr=1e-3)_NCE_0.5ReMSE(0.1,0.0)_seed=214.pth'

    # rezsl model load
    model = build_AttentionNet(dataset_name=opt.dataset)
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=opt.GPU)
    model.load_state_dict(torch.load(attention_model_path)['model'])
    model.eval()

    img_loc = data.test_unseen_loc
    image_files = data.image_files
    image_labels = data.label
    imlist = default_flist_reader(opt, image_files, img_loc, image_labels, opt.dataset)
    test_gzsl(opt, model, imlist, attribute_all, attribute_names, class_names)

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def draw_attention_map(attention, image, attention_path):
    '''
    :param attention: [7, 7]
    :param image: [3, 224, 224]
    :return:
    '''
    image_np = image.permute(1, 2, 0).data.cpu().numpy()

    attention = attention.reshape(1, 1, 7, 7)
    attention = torch.nn.functional.interpolate(attention, scale_factor=32, mode='bilinear')
    attention_np = attention.reshape(224, 224).cuda().data.cpu().numpy()
    vis = show_cam_on_image(image_np, attention_np)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    plt.imsave(attention_path, vis)
    plt.clf()


def test_gzsl(opt, model, imlist, attribute_all, attribute_names,class_names):
    step_num = len(imlist)
    with torch.no_grad():
        for i in range(len(imlist)):
            print("%.d/%.d" % (i, step_num))
            if i < 2701:
                img, label = Image.open(imlist[i][0]), torch.tensor(imlist[i][1])

                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    ])

                img_original = test_transform(img)
                img_transformed = normalize(img_original)
                target_attri = attribute_all[label, :]

                if opt.cuda:
                    img_transformed = img_transformed.cuda()
                    img_original = img_original.cuda()
                    label = label.cuda()
                    target_attri = target_attri.cuda()

                pre_attri, attention = model(x=img_transformed.unsqueeze(0), support_att=attribute_all, getAttention=True) # AttentionMap: [32, 312, 49]
                B, S, N = attention.shape
                H = W = int(N**0.5)

                image_np = img_original.permute(1, 2, 0).data.cpu().numpy()
                save_dir = './Statistics/Attention/%s/%d_%s'%(opt.dataset,i,class_names[label.item()][0][0])
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                plt.imsave(save_dir + '/zzzz_img.png', image_np)
                plt.clf()

                for s in range(S):
                    if target_attri[s]>50.0:
                        attention_specific = attention[0, s, :].view(H, W)
                        attention_path = '%s/%s_%.1f_ReMSE.png'%(save_dir,attribute_names[s], target_attri[s].item())
                        draw_attention_map(attention_specific, img_original, attention_path)

main()