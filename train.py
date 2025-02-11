"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import torch
from tqdm import tqdm
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
import shutil
from logger import Logger
from torchvision.utils import make_grid
from trainers import create_trainer
import numpy as np
from save_remote_gs import init_remote, upload_remote
# from models.networks.sync_batchnorm import DataParallelWithCallback
# from pytorch_fid.fid_model import FIDModel
from pytorch_fid.fid_score import calculate_fid_given_paths, FID_Evaluator
from pathlib import Path
from PIL import Image
from util.util import inverse_transform


# parse options
opt = TrainOptions().parse()

# fid
# fid_model = FIDModel().cuda()
# fid_model.model = DataParallelWithCallback(
#         fid_model.model,
#         device_ids=opt.gpu_ids)


# load remote 
if opt.save_remote_gs is not None:
    init_remote(opt)

# print options to help debugging
# print(' '.join(sys.argv))

# load the dataset
# if opt.dataset_mode_val is not None:
#     dataloader_train, dataloader_val = data.create_dataloader_trainval(opt)
# else:
#     dataloader_train = data.create_dataloader(opt)
#     dataloader_val = None
dataloader_train = data.create_dataloader(opt,'train')
if opt.validation_freq>0:
    dataloader_val = data.create_dataloader(opt,'val')
    fid_eval = FID_Evaluator(32,0,2048,4)

# create trainer for our model
trainer = create_trainer(opt)
model = trainer.pix2pix_model

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader_train))
best_fid = torch.inf

# create tool for visualization
writer = Logger(f"output/{opt.name}")
with open(f"output/{opt.name}/savemodel", "w") as f:
    f.writelines("n")

trainer.save('latest')

def get_psnr(generated, gt):
    generated = inverse_transform(generated)*255
    bsize, c, h, w = gt.shape
    gt = inverse_transform(gt)*255
    mse = ((generated-gt)**2).sum(3).sum(2).sum(1)
    mse /= (c*h*w)
    psnr = 10*torch.log10(255.0*255.0 / (mse+1e-8))
    return psnr.sum().item()

def display_batch(epoch, data_i):
    losses = trainer.get_latest_losses()
    for k,v in losses.items():
        writer.add_scalar(k,v.mean().item(), iter_counter.total_steps_so_far)
    writer.write_console(epoch, iter_counter.epoch_iter, iter_counter.time_per_iter)
    num_print = min(4, data_i['image'].size(0))
    writer.add_single_image('inputs',
            inverse_transform(make_grid(trainer.get_latest_inputs()[:num_print])),
            iter_counter.total_steps_so_far)
    infer_out,inp = trainer.pix2pix_model.forward(data_i, mode='inference')
    vis = inverse_transform(make_grid(inp[:num_print]))
    writer.add_single_image('infer_in',
            vis,
            iter_counter.total_steps_so_far)
    vis = inverse_transform(make_grid(infer_out[:num_print]))
    vis = torch.clamp(vis, 0,1)
    writer.add_single_image('infer_out',
            vis,
            iter_counter.total_steps_so_far)
    generated = trainer.get_latest_generated()
    for k,v in generated.items():
        if v is None:
            continue
        if 'label' in k:
            vis = make_grid(v[:num_print].expand(-1,3,-1,-1))[0]
            writer.add_single_label(k,
                    vis,
                    iter_counter.total_steps_so_far)
        else:
            if v.size(1) == 3:
                vis = inverse_transform(make_grid(v[:num_print]))
                vis = torch.clamp(vis, 0,1)
            else:
                vis = make_grid(v[:num_print])
            writer.add_single_image(k,
                    vis,
                    iter_counter.total_steps_so_far)
    writer.write_html()

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in tqdm(enumerate(dataloader_train, start=iter_counter.epoch_iter),total=len(dataloader_train)-iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        # train discriminator
        if not opt.freeze_D:
            trainer.run_discriminator_one_step(data_i, i)

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, i)

        if iter_counter.needs_displaying():
            display_batch(epoch, data_i)
        if opt.save_remote_gs is not None and iter_counter.needs_saving():
            upload_remote(opt)
        if iter_counter.needs_validation():
            # print('saving the latest model (epoch %d, total_steps %d)' %
            #       (epoch, iter_counter.total_steps_so_far))
            # trainer.save('epoch%d_step%d'%
                    # (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            # iter_counter.record_current_iter()
            # Path(f'temp/gt').mkdir(parents=True, exist_ok=True)
            # Path(f'temp/gen').mkdir(parents=True, exist_ok=True)
            if dataloader_val is not None:
                print("doing validation")
                model.eval()
                num = 0
                psnr_total = 0
                for ii, data_ii in enumerate(dataloader_val):
                    with torch.no_grad():
                        generated,_ = model(data_ii, mode='inference')
                        generated = generated.cpu()
                    gt = data_ii['image']
                    bsize = gt.size(0)
                    psnr = get_psnr(generated, gt)
                    psnr_total += psnr
                    num += bsize
                    fid_eval.add_sample(inverse_transform(generated), inverse_transform(gt))
                    # fid_model.add_sample((generated+1)/2,(gt+1)/2)
                    #Save images
                    # for i in range(bsize):
                    #     img = generated[i]
                    #     img = inverse_transform(img)
                    #     if np.array(img).min()<0 or np.array(img).max()>255:
                    #         print(img.min(), img.max())
                    #     img.save(f'temp/gen/{num+i}.png')
                    #     img = gt[i]#.permute(1,2,0)
                    #     img = inverse_transform(img)#.permute(1,2,0)
                    #     img.save(f'temp/gt/{num+i}.png')
                psnr_total /= num
                # fid = fid_model.calculate_activation_statistics()
                # fid = calculate_fid_given_paths(('temp/gt', 'temp/gen'), 32, 0, 2048, 4)
                # fid_eval.generated = torch.cat(fid_eval.generated,0)
                # fid_eval.real = torch.cat(fid_eval.real,0)
                # print(len(fid_eval.generated),len(fid_eval.real))
                # print(type(fid_eval.generated),type(fid_eval.real))
                fid = fid_eval.get_fid()
                fid_eval.clear()
                if fid < best_fid:
                    best_fid = fid
                    trainer.save('best')
                print(f"FID: {fid}")
                #Remove temp folder
                # shutil.rmtree('temp')
                writer.add_scalar("val.fid", fid, iter_counter.total_steps_so_far)
                writer.write_scalar("val.fid", fid, iter_counter.total_steps_so_far)
                writer.add_scalar("val.psnr", psnr_total, iter_counter.total_steps_so_far)
                writer.write_scalar("val.psnr", psnr_total, iter_counter.total_steps_so_far)
                writer.write_html()
                model.train()
    trainer.update_learning_rate(epoch)
    if epoch != 0 and epoch % 3 == 0 and opt.dataset_mode_train == 'cocomaskupdate':
        dataloader_train.dataset.update_dataset()
    iter_counter.record_epoch_end()

print('Training was successfully finished.')
