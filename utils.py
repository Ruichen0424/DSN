############################# Load model #############################
def get_model(args):
    if args.model == 'msnet18':
        from Network.MSNet import resnet18
        model = resnet18(num_classes=args.num_classes, imagenet=args.dataset.lower()=='imagenet', T=args.timestep)
    elif args.model == 'msnet34':
        from Network.MSNet import resnet34
        model = resnet34(num_classes=args.num_classes, imagenet=args.dataset.lower()=='imagenet', T=args.timestep)
    elif args.model == 'spikedriventransformer':
        from Network.SpikeDrivenTransformer import spikedriventransformer
        model = spikedriventransformer(
            imagenet=args.dataset.lower()=='imagenet', T=args.timestep,
            embed_dims=args.dim, num_heads=args.num_heads,
            num_classes=args.num_classes, depths=args.depths,
        )
    else:
        raise ValueError(f'Unknown model {args.model}')
    return model


############################# Get Dataset function #############################
from torchvision import datasets
from timm.data import create_dataset, create_loader

def get_loader(args, data_config, train_interpolation, num_aug_splits=0, collate_fn=None):
    if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100' or args.dataset.lower() == 'imagenet':
        dataset = f'torch/{args.dataset}' if args.dataset.lower() != 'imagenet' else 'torch/folder'
        dataset_train = create_dataset(dataset, root=args.data_dir, split=args.train_split, is_training=True, batch_size=args.batch_size, repeats=args.epoch_repeats)
        dataset_eval = create_dataset(dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader
        )

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.val_batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )
        return dataset_train, dataset_eval, loader_train, loader_eval
    elif args.dataset.lower() == 'dvsgesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        Train_data = DVS128Gesture(root=args.dataset_path, train=True, data_type='frame', frames_number=args.timestep, split_by='number')
        Test_data = DVS128Gesture(root=args.dataset_path, train=False, data_type='frame', frames_number=args.timestep, split_by='number')
        return Train_data, Test_data


from Network.DNeuro import DLIFNode
def set_d_threshold(model, D_threshold, store_D_seq=False, store_v_seq=False):
    for m in model.modules():
        if isinstance(m, DLIFNode):
            m.D_threshold = D_threshold
            m.store_D_seq = store_D_seq
            m.store_v_seq = store_v_seq