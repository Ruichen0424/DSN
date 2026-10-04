CUDA_VISIBLE_DEVICES=0 python train.py -c cifar10_ms18.yml
CUDA_VISIBLE_DEVICES=0 python train.py -c cifar10_sdt.yml

CUDA_VISIBLE_DEVICES=0 python train.py -c cifar100_ms18.yml
CUDA_VISIBLE_DEVICES=0 python train.py -c cifar100_sdt.yml

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py -c imagenet_ms18.yml
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py -c imagenet_ms34.yml
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py -c imagenet_sdt.yml