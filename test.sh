python test.py \
	--mixing 0 \
	--batchSize 1 \
	--nThreads 1 \
	--name comod_imagenet \
	--dataset_mode testimage \
	--load_size 256 \
	--crop_size 256 \
	--z_dim 512 \
	--model comod \
	--netG comodgan \
        --which_epoch co-mod-gan-ffhq-9-025000 \
		--val_annfile /raid/student/2021/ai21btech11005/PartImageNet/annotations/test/test.json \
	--load_pretrained_g_ema /raid/student/2021/ai21btech11005/co-mod-gan-pytorch/checkpoints/comod_imagenet/best_net_G_ema.pth
	${EXTRA} \
	# --image_dir ./ffhq_debug/images \
	# --mask_dir ./ffhq_debug/masks \
    #     --output_dir ./comod_imagenet_eval \ 
	