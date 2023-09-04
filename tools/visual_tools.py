import torch
import numpy as np
from PIL import Image

def prepare_attri_label(attribute, classes):
    # print("attribute.shape", attribute.shape)
    classes_dim = classes.size(0)
    attri_dim = attribute.shape[1]
    output_attribute = torch.FloatTensor(classes_dim, attri_dim)
    for i in range(classes_dim):
        output_attribute[i] = attribute[classes[i]]
    return output_attribute

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(opt, image_files, img_loc, image_labels, dataset):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    image_files = image_files[img_loc]
    image_labels = image_labels[img_loc]
    for image_file, image_label in zip(image_files, image_labels):
        if dataset == 'CUB':
            image_file = opt.image_root + image_file[0].split("MSc/CUB_200_2011")[1]
        elif dataset == 'AWA1':
            image_file = opt.image_root + image_file[0].split("databases/")[1]
        elif dataset == 'AWA2':
            image_file = opt.image_root + 'AWA2/Animals_with_Attributes2/JPEGImages' + image_file[0].split("JPEGImages")[1]
        elif dataset == 'SUN':
            image_file =  opt.image_root + image_file[0].split("data/")[1]
        else:
            exit(1)
        imlist.append((image_file, int(image_label)))
    return imlist

class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, opt, data_inf=None, transform=None, target_transform=None, dataset=None,
                 flist_reader=default_flist_reader, loader=default_loader, image_type=None, select_num=None):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        if image_type == 'test_unseen_small_loc':
            print('test_unseen_small_loc')
            self.img_loc = data_inf.test_unseen_small_loc
        elif image_type == 'test_unseen_loc':
            print('test_unseen_loc')
            self.img_loc = data_inf.test_unseen_loc
        elif image_type == 'test_seen_loc':
            print('test_seen_loc')
            self.img_loc = data_inf.test_seen_loc
        elif image_type == 'trainval_loc':
            print('trainval_loc')
            self.img_loc = data_inf.trainval_loc
        elif image_type == 'train_loc':
            print('train_loc')
            self.img_loc = data_inf.train_loc
        elif image_type == 'test_loc':
            print('test_loc')
            self.img_loc = np.append(data_inf.test_seen_loc,data_inf.test_unseen_loc)
        else:
            print('else')
            try:
                sys.exit(0)
            except:
                print("choose the image_type in ImageFileList")

        if select_num != None:
            # select_num is the number of images that we want to use
            # shuffle the image loc and choose #select_num images
            np.random.shuffle(self.img_loc)
            self.img_loc = self.img_loc[:select_num]

        self.image_files = data_inf.image_files
        self.image_labels = data_inf.label
        self.dataset = dataset
        self.imlist = flist_reader(opt, self.image_files, self.img_loc, self.image_labels, self.dataset)
        self.allclasses_name = data_inf.allclasses_name

        self.image_labels = self.image_labels[self.img_loc]
        label, idx = np.unique(self.image_labels, return_inverse=True)
        self.image_labels = torch.tensor(idx)

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, impath

    def __len__(self):
        num = len(self.imlist)
        return num