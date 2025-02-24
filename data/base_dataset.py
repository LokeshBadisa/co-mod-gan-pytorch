"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def bbox_transform(opt, params, bbox, size, force_flip=False):
    w,h = size
    if opt.isTrain and (not opt.no_flip or force_flip):
        if params['flip']:
            bbox[0] = w-bbox[0]-bbox[2]
    rate_h = rate_w = 1
    if 'resize' in opt.preprocess_mode:
        rate_h = float(opt.load_size)/h
        rate_w = float(opt.load_size)/w
    elif 'scale_width' in opt.preprocess_mode:
        rate_w = rate_h = float(opt.load_size)/w
    elif 'scale_shortside' in opt.preprocess_mode:
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        if (ss == opt.load_size):
            rate_w = rate_h = 1
        ls = int(opt.load_size * ls / ss)
        nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
        rate_h = float(nh)/h
        rate_w = float(nw)/w
    if opt.preprocess_mode == 'fixed':
        rate_w = float(opt.crop_size)/w
        ht = round(opt.crop_size / opt.aspect_ratio)
        rate_h = float(opt.crop_size)/ht

    bbox[0] *= rate_w
    bbox[2] *= rate_w
    bbox[1] *= rate_h
    bbox[3] *= rate_h
    w *= rate_w
    h *= rate_h

    if 'crop' in opt.preprocess_mode:
        x,y = params['crop_pos']
        bbox[0] -= x
        bbox[1] -= y
        w -= x
        h -= y
        y2 = bbox[1]+bbox[3]
        x2 = bbox[0]+bbox[2]
        y2 = min(opt.crop_size,y2)
        x2 = min(opt.crop_size,x2)
        x1 = max(0,bbox[0])
        y1 = max(0,bbox[1])
        bbox[0] = x1
        bbox[1] = y1
        bbox[2] = x2-x1
        bbox[3] = y2-y1
    if opt.preprocess_mode == 'none':
        base = 32
        _h = int(round(oh / base) * base)
        _w = int(round(ow / base) * base)
        rate_h = float(_h)/h
        rate_w = float(_w)/w
        bbox[0] *= rate_w
        bbox[2] *= rate_w
        bbox[1] *= rate_h
        bbox[3] *= rate_h
    return bbox


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True, force_flip=False):
    transform_list = []
    if opt.isTrain and (not opt.no_flip or force_flip):
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))]
        # transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
        #                                         (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    # return transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
