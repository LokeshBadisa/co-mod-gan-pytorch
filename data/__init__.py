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
        self.maskroot = Path(opt.val_annfile).parent.parent/Path(opt.val_annfile).stem
        self.imageroot = self.maskroot.parent.parent/'images'/ Path(opt.val_annfile).stem
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
                
        # self.deny_indices = [21, 85, 145, 380, 969, 977, 1158]  # For val
        self.deny_indices = [65, 347, 533, 657, 728, 945, 969, 1061, 1613, 1630, 1738, 1819, 1906, 1928, 2001, 2219, 2254, 2406]  #For test
        self.allow_indices = [i for i in range(len(self.data.dataset['images'])) if i not in self.deny_indices]

    def __len__(self):
        return len(self.allow_indices)
    
    def __getitem__(self, idx):
        idx = self.allow_indices[idx]
        file_name = Path(self.data.loadImgs(idx)[0]['file_name'])
        image = Image.open(str(self.imageroot/file_name)).convert('RGB')
        # try:
        mask = multiload(self.data, self.data.getAnnIds(imgIds=idx))
        # except:
        #     print("Error loading index ", idx)
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