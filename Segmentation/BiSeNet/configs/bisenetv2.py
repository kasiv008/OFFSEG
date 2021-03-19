
## bisenetv2
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start = 5e-2,
    weight_decay=5e-4,
    warmup_iters = 1000,
    max_iter = 150000,
    im_root='/media/storage/data/moonlab/OffRoad/Dataset',
    train_im_anns='/media/storage/data/moonlab/OffRoad/Dataset/train.txt',
    val_im_anns='/media/storage/data/moonlab/OffRoad/Dataset/val.txt',
    scales=[0.25, 2.],
    cropsize=[1024,640],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='/media/storage/data/moonlab/OffRoad/Dataset/res',
)
