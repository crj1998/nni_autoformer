
python evaluate.py --weights /root/rjchen/workspace/autoformer/weights/supernet20220822.pth --datapath ../../data/ImageNet/val

# Random Search
python search.py --weights /root/rjchen/workspace/autoformer/weights/supernet20220822.pth --datapath ../../data/ImageNet/val
# Evolution Search
python search.py --weights weights/supernet-tiny.pth --datapath path/to/cifar10 --evolution --evolution-sample-size 100 --evolution-population-size 50 --evolution-cycles 1000

# CUDA_VISIBLE_DEVICES=0,1,2,3 python scratch.py --gpus 4 --datapath path/to/imagenet --batch_size 256 --warmup 20 --epochs 500 --learning_rate 0.001
CUDA_VISIBLE_DEVICES=0,1,2,3 python scratch.py --gpus 4 --datapath /mnt/imagenet/all --batch_size 256 --warmup 0 --epochs 1 --learning_rate 0.001

# CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --weights weights/supernet-tiny.pth --datapath path/to/cifar10 --epochs 100 --warmup 5 --gpus 4 --learning_rate 0.0005