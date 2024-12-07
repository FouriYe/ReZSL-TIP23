from os.path import join

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing

from .random_dataset import RandDataset
from .episode_dataset import EpiDataset, CategoriesSampler, DCategoriesSampler
from .test_dataset import TestDataset

from .transforms import data_transform

import scipy.io as sio
import copy

class ImgDatasetParam(object):
    DATASETS = {
        "imgroot": '/Dataset',
        "dataroot": '/Dataset/xlsa17/data',
        "image_embedding": 'res101',
        "class_embedding": 'att'
    }

    @staticmethod
    def get(dataset):
        attrs = ImgDatasetParam.DATASETS
        attrs["imgroot"] = join(attrs["imgroot"], dataset)
        args = dict(
            dataset=dataset
        )
        args.update(attrs)
        return args

def build_dataloader(cfg, is_distributed=False):

    args = ImgDatasetParam.get(cfg.DATASETS.NAME)
    imgroot = args['imgroot']
    dataroot = args['dataroot']
    image_embedding = args['image_embedding']
    class_embedding = args['class_embedding']
    dataset = args['dataset']

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

    img_files =np.squeeze(matcontent['image_files'])
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        if dataset=='CUB':
            img_path = imgroot[:-4] + img_path.split("MSc/CUB_200_2011")[1]
        elif dataset=='AWA2':
            img_path = imgroot[:-4] + 'AWA2/Animals_with_Attributes2/JPEGImages' + img_path.split("JPEGImages")[1]
        elif dataset=='SUN':
            img_path = join(imgroot, img_path.split("SUN/")[1])
        new_img_files.append(img_path)

    new_img_files = np.array(new_img_files)
    label = matcontent['labels'].astype(int).squeeze() - 1


    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    trainvalloc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    cls_name = matcontent['allclasses_names']
    if cfg.DATASETS.SEMANTIC_TYPE =="GBU":
        if cfg.DATASETS.SEMANTIC == "normalized":
            att_name = 'att'
        elif cfg.DATASETS.SEMANTIC == "original":
            att_name = 'original_att'
        else:
            print('unrecognized SEMANTIC')
            att_name = 'att'
        attribute = matcontent[att_name].T
    else:
        print('unrecognized SEMANTIC TYPE')

    train_img = new_img_files[trainvalloc]
    train_label = label[trainvalloc].astype(int)
    train_att = attribute[train_label]
    train_id, idx = np.unique(train_label, return_inverse=True)
    train_att_unique = attribute[train_id]
    train_clsname = cls_name[train_id]

    num_train = len(train_id)
    train_label = idx
    train_id = np.unique(train_label)

    test_img_unseen = new_img_files[test_unseen_loc]
    test_label_unseen = label[test_unseen_loc].astype(int)
    test_id, idx = np.unique(test_label_unseen, return_inverse=True)
    att_unseen = attribute[test_id]
    test_clsname = cls_name[test_id]
    test_label_unseen = idx + num_train
    test_id = np.unique(test_label_unseen)

    train_test_att = np.concatenate((train_att_unique, att_unseen))
    train_test_id = np.concatenate((train_id, test_id))

    test_img_seen = new_img_files[test_seen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    _, idx = np.unique(test_label_seen, return_inverse=True)
    test_label_seen = idx

    att_unseen = torch.from_numpy(att_unseen).float()
    test_label_seen = torch.tensor(test_label_seen)
    test_label_unseen = torch.tensor(test_label_unseen)
    train_label = torch.tensor(train_label)
    att_seen = torch.from_numpy(train_att_unique).float()

    att_all = torch.from_numpy(attribute).float()
    res = {
        'train_label': train_label,
        'train_att': train_att,
        'test_label_seen': test_label_seen,
        'test_label_unseen': test_label_unseen,
        'att_unseen': att_unseen,
        'att_seen': att_seen,
        'att_all': att_all,
        'train_id': train_id,
        'test_id': test_id,
        'train_test_id': train_test_id,
        'train_clsname': train_clsname,
        'test_clsname': test_clsname
    }

    # train dataloader
    ways = cfg.DATASETS.WAYS
    shots = cfg.DATASETS.SHOTS
    data_aug_train = cfg.SOLVER.DATA_AUG
    img_size = cfg.DATASETS.IMAGE_SIZE
    transforms = data_transform(data_aug_train, size=img_size)

    if cfg.DATALOADER.MODE == 'random':
        dataset = RandDataset(train_img, train_att, train_label, transforms)

        if not is_distributed:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
            batch = ways*shots
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch, drop_last=True)
            tr_dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                num_workers=8,
                batch_sampler=batch_sampler,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            batch = ways * shots
            tr_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=8)


    elif cfg.DATALOADER.MODE == 'episode':
        n_batch = cfg.DATALOADER.N_BATCH
        ep_per_batch = cfg.DATALOADER.EP_PER_BATCH
        dataset = EpiDataset(train_img, train_att, train_label, transforms)
        if not is_distributed:
            sampler = CategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        else:
            sampler = DCategoriesSampler(
                train_label,
                n_batch,
                ways,
                shots,
                ep_per_batch
            )
        tr_dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)

    data_aug_test = cfg.TEST.DATA_AUG
    transforms = data_transform(data_aug_test, size=img_size)
    test_batch_size = cfg.TEST.IMS_PER_BATCH

    if not is_distributed:
        # test unseen dataloader
        tu_data = TestDataset(test_img_unseen, att_unseen, test_label_unseen, transforms, atts_offset=-1*num_train)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)

        # test seen dataloader
        ts_data = TestDataset(test_img_seen, att_seen, test_label_seen, transforms)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
    else:
        # test unseen dataloader
        tu_data = TestDataset(test_img_unseen, att_unseen, test_label_unseen, transforms, atts_offset=-1*num_train)
        tu_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tu_data, shuffle=False)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, sampler=tu_sampler,
            num_workers=4, pin_memory=False)

        # test seen dataloader
        ts_data = TestDataset(test_img_seen, att_seen, test_label_seen, transforms)
        ts_sampler = torch.utils.data.distributed.DistributedSampler(dataset=ts_data, shuffle=False)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, sampler=ts_sampler,
            num_workers=4, pin_memory=False)

    return tr_dataloader, tu_loader, ts_loader, res


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.ntrain_class = self.seenclasses.size(0)
        self.train_cls_num = self.seenclasses.shape[0]
        self.feature_dim = self.train_feature.shape[1]
        # # The following are copied from GBU
        self.feature_dim = self.train_feature.shape[1]  # train_feature是横着的
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.train_cls_num = self.seenclasses.shape[0]
        if opt.gzsl:
            self.test_cls_num = self.seenclasses.shape[0] + self.unseenclasses.shape[0]
        else:
            self.test_cls_num = self.unseenclasses.shape[0]

    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()]
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()]
        val_unseen_loc = fid['val_unseen_loc'][()]
        test_seen_loc = fid['test_seen_loc'][()]
        test_unseen_loc = fid['test_unseen_loc'][()]
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc]
            self.train_label = label[trainval_loc]
            self.test_unseen_feature = feature[test_unseen_loc]
            self.test_unseen_label = label[test_unseen_loc]
            self.test_seen_feature = feature[test_seen_loc]
            self.test_seen_label = label[test_seen_loc]
        else:
            self.train_feature = feature[train_loc]
            self.train_label = label[train_loc]
            self.test_unseen_feature = feature[val_unseen_loc]
            self.test_unseen_label = label[val_unseen_loc]

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")

        feature = matcontent['features'].T
        self.label = matcontent['labels'].astype(int).squeeze() - 1
        self.image_files = matcontent['image_files'].squeeze()
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        self.trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        if opt.dataset == 'CUB':
            self.train_loc = matcontent['train_loc'].squeeze() - 1
            self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1

        self.test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.allclasses_name = matcontent['allclasses_names']
        if opt.SEMANTIC == 'normalized':
            self.attribute = torch.from_numpy(matcontent['att'].T).float()
        else:
            self.attribute = torch.from_numpy(matcontent['original_att'].T).float()

        self.ori_attribute = torch.from_numpy(matcontent['original_att'].T).float()

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[self.trainval_loc])
                _test_seen_feature = scaler.transform(feature[self.test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[self.test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[self.trainval_loc]).float()
                self.train_label = torch.from_numpy(self.label[self.trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[self.test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(self.label[self.test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[self.test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(self.label[self.test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[self.train_loc]).float()
            self.train_label = torch.from_numpy(self.label[self.train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[self.val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(self.label[self.val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

        self.att_dim = self.attribute.size(1)
        self.feature_dim = self.test_seen_feature.size()[1]
        self.train_class = self.seenclasses
        self.train_att = self.attribute[self.seenclasses]  # tensor
        self.test_att = self.attribute[self.unseenclasses]  # tensor
        self.train_cls_num = 150
        self.test_cls_num = 50
        self.class_names = matcontent['allclasses_names']


    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att

def map_label(label, classes):
    # original label -> seen or unseen label index
    mapped_label = torch.LongTensor(len(label))
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label
