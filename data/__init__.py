"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# import importlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from data.base_dataset import BaseDataset
from data.base_dataset import get_params, get_transform
from torchvision.datasets import ImageNet
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
from torchvision import transforms

def multiload(data, L):
    ans = data.annToMask(data.loadAnns(L[0])[0])
    for i in range(1,len(L)):
        ans += data.annToMask(data.loadAnns(L[i])[0])
    return ans

    
class PartImageNetDataset(Dataset):
    def __init__(self, opt):
        self.maskroot = Path(opt.val_annfile).parent/Path(opt.val_annfile).stem
        self.imageroot = self.maskroot.parent/ Path(opt.val_annfile).stem
        self.data = COCO(opt.val_annfile)   
        transform_list = [
                transforms.Resize((opt.crop_size, opt.crop_size), 
                    interpolation=Image.BICUBIC),
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))
                # transforms.Normalize((0.5, 0.5, 0.5),
                #                     (0.5, 0.5, 0.5))
                ]
        self.image_transform = transforms.Compose(transform_list)
        self.mask_transform = transforms.Compose([
            transforms.Resize((opt.crop_size, opt.crop_size),interpolation=Image.BICUBIC),
            transforms.ToTensor()
            ])     

        if Path(opt.val_annfile).stem == 'val':
            self.deny_indices = [0, 81, 258, 263, 300, 313, 560, 572, 1150,\
                             1169, 1211, 1494, 1870, 2018, 2347, 2362,\
                             2402, 2683, 2726] #val
        elif Path(opt.val_annfile).stem == 'test':
            self.deny_indices = [213, 291, 455, 494, 515, 516, 520, 592, 727,\
                             761, 1576, 1816, 2415, 2752, 2893, 2995, 3025,\
                             3182, 3278, 3336, 3349, 3399, 3453, 3459, 3616,\
                             3639, 3779, 3837, 4040, 4190, 4435, 4441, 4477,\
                             4479, 4510, 4576, 4597]   #test
        else:
            self.deny_indices = [231, 252, 511, 841, 1064, 1078, 1273, 1288,\
                             1323, 1393, 1450, 1617, 1725, 1876, 1960,\
                             2018, 2091, 2104, 2457, 2509, 2563, 2727,\
                             2816, 2854, 2867, 3078, 3087, 3165, 3285,\
                             3327, 3331, 3712, 3738, 3947, 4052, 4066,\
                             4089, 4101, 4116, 4174, 4270, 4348, 4396,\
                             4502, 4614, 4629, 4770, 4886, 5032, 5114,\
                             5356, 5377, 5469, 5669, 6331, 6487, 6693,\
                             6901, 6918, 7130, 7214, 7290, 7377, 7457,\
                             7593, 7708, 7811, 7836, 7936, 8085, 8247,\
                             8368, 8719, 8763, 8895, 9214, 9245, 9278,\
                             10015, 10228, 10896, 10992, 11127, 11156,\
                             11170, 11177, 11362, 11775, 11888, 11906,\
                             11942, 12099, 12109, 12124, 12182, 12327,\
                             12417, 12441, 12516, 12892, 12925, 12951,\
                             12965, 13171, 13203, 13280, 13336, 13368,\
                             13485, 13539, 13564, 13603, 13744, 13751,\
                             13793, 13833, 14022, 14279, 14421, 14688,\
                             14757, 14813, 14868, 14914, 14994, 14995,\
                             15018, 15161, 15293, 15297, 15365, 15465,\
                             15574, 16019, 16031, 16245, 16496] #train
        self.allow_indices = [i for i in range(len(self.data.dataset['images'])) if i not in self.deny_indices]        

    def __len__(self):
        return len(self.allow_indices)
    
    def __getitem__(self, idx):
        idx = self.allow_indices[idx]
        file_name = Path(self.data.loadImgs(idx)[0]['file_name'])
        folder = Path(self.data.loadImgs(idx)[0]['file_name'].split('_')[0])
        image = Image.open(str(self.imageroot/folder/file_name)).convert('RGB')        
        try:
            mask = multiload(self.data, self.data.getAnnIds(imgIds=idx))        
        except:
            print('Error Loading', idx)
            return None
        mask = np.where(mask>0,1,0)
        mask = Image.fromarray((mask*255).astype(np.uint8))
        sample = {'image': self.image_transform(image), 'mask': self.mask_transform(mask),'path':str(file_name)}
        return sample
    

class FilteredImageNet(Dataset):
    def __init__(self, imagenet_dir, classes_file, partition, opt):
        self.classes = []
        self.opt = opt
        with open(classes_file, 'r') as f:
            for line in f:
                self.classes.append(line.strip())        
        self.train_dataset = ImageNet(root=imagenet_dir, split=partition)
        self.classes = [self.train_dataset.wnid_to_idx[i] for i in self.classes]
        self.filtered_indices = [i for i, target in enumerate(self.train_dataset.targets) if target in self.classes]
        self.filtered_indices_map = {Path(self.train_dataset.imgs[i][0]).stem:i for i in self.filtered_indices}

        partinet = PartImageNetDataset(opt)
        for i in range(len(partinet)):
            self.filtered_indices_map.pop(Path(partinet.data.dataset['images'][i]['file_name']).stem, None)
                        
        self.filtered_indices = [self.filtered_indices_map[k] for k in self.filtered_indices_map]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        image = self.train_dataset[self.filtered_indices[idx]][0] 
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        return {'image':transform_image(image)}

class NativeImageNet(Dataset):
    def __init__(self, imagenet_dir, partition, opt):
        self.classes = []
        self.opt = opt
        # with open(classes_file, 'r') as f:
        #     for line in f:
        #         self.classes.append(line.strip())        
        self.train_dataset = ImageNet(root=imagenet_dir, split=partition)
        # self.classes = [self.train_dataset.wnid_to_idx[i] for i in self.classes]
        # self.filtered_indices = [i for i, target in enumerate(self.train_dataset.targets) if target in self.classes]
        # self.filtered_indices_map = {Path(self.train_dataset.imgs[i][0]).stem:i for i in self.filtered_indices}

        # partinet = PartImageNetDataset(opt)
        # for i in range(len(partinet)):
        #     self.filtered_indices_map.pop(Path(partinet.data.dataset['images'][i]['file_name']).stem, None)
                        
        # self.filtered_indices = [self.filtered_indices_map[k] for k in self.filtered_indices_map]

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        # image = self.train_dataset[self.filtered_indices[idx]][0] 
        image = self.train_dataset[idx][0]
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        return {'image':transform_image(image)}

# def find_dataset_using_name(dataset_name):
#     # Given the option --dataset [datasetname],
#     # the file "datasets/datasetname_dataset.py"
#     # will be imported. 
#     dataset_filename = "data." + dataset_name + "_dataset"
#     datasetlib = importlib.import_module(dataset_filename)

#     # In the file, the class called DatasetNameDataset() will
#     # be instantiated. It has to be a subclass of BaseDataset,
#     # and it is case-insensitive.
#     dataset = None
#     target_dataset_name = dataset_name.replace('_', '') + 'dataset'
#     for name, cls in datasetlib.__dict__.items():
#         if name.lower() == target_dataset_name.lower() \
#            and issubclass(cls, BaseDataset):
#             dataset = cls
            
#     if dataset is None:
#         raise ValueError("In %s.py, there should be a subclass of BaseDataset "
#                          "with class name that matches %s in lowercase." %
#                          (dataset_filename, target_dataset_name))

#     return dataset


# def get_option_setter(dataset_name):    
#     dataset_class = find_dataset_using_name(dataset_name)
#     return dataset_class.modify_commandline_options


def create_dataloader(opt,partition):
    if partition == 'train':
        dataset = FilteredImageNet(imagenet_dir=opt.imagenet_dir, classes_file=opt.classes_list, partition=partition, opt=opt)
        # dataset = NativeImageNet(imagenet_dir=opt.imagenet_dir, partition=partition, opt=opt)
    elif partition == 'val':
        dataset = PartImageNetDataset(opt)
    print(dataset.__len__())
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=True
    )
    return dataloader