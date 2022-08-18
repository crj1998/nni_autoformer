# NNI implementation of AutoFormer
This is a simple example that demonstrates how to use NNI to implement a [AutoFormer: Searching Transformers for Visual Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_AutoFormer_Searching_Transformers_for_Visual_Recognition_ICCV_2021_paper.html) search space. The official implementation code can be found [here](https://github.com/microsoft/Cream/tree/main/AutoFormer). 

This example pipline include four parts :

1. Evaluate the pre-searched model on valid dataset.
2. Evolution search the best sub-net on the pretrained model space (super-net).
3. Fine-tuning the ImageNet-1k pretrained supernet to cifar10.
4. Train the supernet from scratch on ImageNet-1k.


## Quick Start

### prepare dataset
Train from scratch requires ImageNet-1k dataset, fine-tune requires CIFAR-10. Prepare teh dataset at first.

### Evaluate
`hub.AutoformerSpace` provide the function `load_searched_model(name)` to obtain pre-searched sub-model architecture and weights.
Use following srcipt to evaluate valid dataset accuracy on the pre-searched model.

```
python evaluate.py --datapath path/to/imagenet/val --weights path/to/weights.pth
# python evaluate.py --datapath ../../data/ImageNet/val --weights /root/rjchen/workspace/devnni/weights/tiny.pth
```

NNI implementation result.

| Size  |        name        | Params. | Acc@Top-1 | Acc@Top-5 |
|-------|--------------------|---------|-----------|-----------|
| Tiny  | `autoformer-tiny`  |   5.8M  |  75.31    |   92.69   |
| Small | `autoformer-small` |  22.9M  |  81.67    |   95.72   |
| Base  | `autoformer-base`  |  53.7M  |  82.39    |   95.74   |


### Evolution Search
With a well-trained supernet, we need to perform Evolution strategy to search the best subnet under optional latency constrants. 
```
python search.py --weights weights/supernet-tiny.pth --datapath path/to/cifar10 --evolution-sample-size 100 --evolution-population-size 50 --evolution-cycles 10000
```

### Training
#### Train from scratch on ImageNet-1k
Train a supernet on the ImageNet from scratch. 

```
python scratch.py --gpus 4 --datapath path/to/imagenet --batch_size 256 --warmup 20 --epochs 500 --learning_rate 0.001
# CUDA_VISIBLE_DEVICES=0,1,2,3 python scratch.py --gpus 4 --datapath path/to/imagenet --batch_size 256 --warmup 20 --epochs 500 --learning_rate 0.001
```

Note: When training from scratch on ImageNet-1K with 4 Tesla V100s, it consumes about 1days per 100 epochs. 

#### Fine-tune on CIFAR10
With a ImageNet-1k pretrained supernet, we can fine-tuning it to CIFAR10. 
```
python finetune.py --gpus 4 --weights weights/supernet-tiny.pth --datapath path/to/cifar10 --epochs 100 --warmup 5 --learning_rate 0.0005
# CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --weights weights/supernet-tiny.pth --datapath path/to/cifar10 --epochs 100 --warmup 5 --gpus 4 --learning_rate 0.0005
```


### Tools: convert 
Convert Autoformer official pretrained weight to nni format.

```
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-tiny.pth -O official-supernet-tiny.pth
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-small.pth -O official-supernet-small.pth
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-base.pth -O official-supernet-base.pth
```