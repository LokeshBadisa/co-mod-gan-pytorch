export CXX="g++"
python train.py \
	--batchSize 32 \
	--nThreads 4 \
	--name comod_imagenet7 \
	--load_size 256 \
	--crop_size 256 \
	--z_dim 512 \
	--validation_freq 10000 \
	--niter 125 \
	--niter_decay 0 \
	--lr 0.002 \
	--dataset_mode trainimage \
	--trainer stylegan2 \
	--model comod \
	--netG comodgan \
	--netD comodgan \
	--no_l1_loss \
	--no_vgg_loss \
	--preprocess_mode scale_shortside_and_crop \
	--imagenet_dir /raid/student/2021/ai21btech11005/Imagenet \
	--classes_list ./PIN_classes.txt\
	--load_pretrained_g_ema /raid/student/2021/ai21btech11005/co-mod-gan-pytorch/checkpoints/comod_imagenet1/best_net_G_ema.pth \
	--load_pretrained_g /raid/student/2021/ai21btech11005/co-mod-gan-pytorch/checkpoints/comod_imagenet1/best_net_G.pth \
	--load_pretrained_d /raid/student/2021/ai21btech11005/co-mod-gan-pytorch/checkpoints/comod_imagenet1/best_net_D.pth \
	--save_epoch_freq 10 \
	--gpu_id 7,8,9,10,11,12,13,14 \
	--val_annfile /raid/student/2021/ai21btech11005/PartImageNet/annotations/test/test.json \
	$EXTRA
	# --train_image_dir ./datasets/places2sample1k_val/places2samples1k_crop256 \
	# --train_image_list ./datasets/places2sample1k_val/files.txt \
	# --train_image_postfix '.jpg' \
	# --val_image_dir ./datasets/places2sample1k_val/places2samples1k_crop256 \
	# --val_image_list ./datasets/places2sample1k_val/files.txt \
	# --val_mask_dir ./datasets/places2sample1k_val/places2samples1k_256_mask_square128 \
	# --dataset_mode_train trainimage \
	# --dataset_mode_val valimage \
	# --load_pretrained_g_ema ./checkpoints/comod-places-512/co-mod-gan-places2-050000_net_G_ema.pth \