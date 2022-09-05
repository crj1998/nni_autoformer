# NNI implementation of AutoFormer
This is a simple example that demonstrates how to use NNI to implement a [AutoFormer: Searching Transformers for Visual Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_AutoFormer_Searching_Transformers_for_Visual_Recognition_ICCV_2021_paper.html) search space. The official implementation code can be found [here](https://github.com/microsoft/Cream/tree/main/AutoFormer). 

This example pipline include four parts :

1. Evaluate the pre-searched model on valid dataset.
2. Evolution search the best sub-net on the pretrained model space.
3. Fine-tuning the ImageNet-1k pretrained model space to cifar10.
4. Train the supernet from scratch on ImageNet-1k.

> TL;DR: For advanced users, please refer to [Tutotial](tutorials.md) for more implentation details.

## Quick Start
### Prepare dataset
[ImageNet-1k](https://www.image-net.org/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) are required in this example. Prepare the dataset at first.

### Evaluate
`hub.AutoformerSpace` provide the interface `load_searched_model(name)` to obtain the pre-searched subnet architecture and optional pretrained weights. 

```python
model = AutoformerSpace.load_searched_model('autoformer-tiny')
acc = valid(model, dataloader)
```

Use following srcipt to evaluate valid dataset accuracy on the pre-searched model.

```bash
# --name:autoformer size. must be one of `tiny`, `small` and `base`.
# --weights: subnet weights. If not specified, the weights provided by NNI are used
python evaluate.py --datapath path/to/imagenet/val --name tiny --weights path/to/subnet.pth
```

NNI implementation reproduce result. It is consistent with the results of the paper.

| Size  |        name        | Params. | Acc@Top-1 | Acc@Top-5 |
|-------|--------------------|---------|-----------|-----------|
| Tiny  | `autoformer-tiny`  |   5.8M  |  75.31    |   92.69   |
| Small | `autoformer-small` |  22.9M  |  81.67    |   95.72   |
| Base  | `autoformer-base`  |  53.7M  |  82.39    |   95.74   |


### Search
With a well-trained supernet, you may need to perform Evolution strategy to search the best subnet under the optional latency constrant. During the evaluation, the performance of each subnet is independently evaluated. NNI's engine evaluates multiple subnetworks simultaneously based on the amount of computing resources. 

Use following srcipt to search the best sub-net on the well-trained model space.

```bash
# Random Search
python search.py --weights path/to/supernet.pth --datapath path/to/imagenet
# Evolution Search
python search.py --weights path/to/supernet.pth --datapath path/to/cifar10 --evolution --evolution-sample-size 100 --evolution-population-size 50 --evolution-cycles 1000
```


### Training
Training from scratch or finetune share the same way except for some hyper-parameters. You can use the evaluator `Classification` provided in `nni.nas.evaluator.pytorch.lightning` directly. However, this build-in evaluator has limited functionality. To replicate the training process in the paper, you need to customize a evaluator to support operations such as data augmentation and learning rate scheduling. The evaluator is in fact a `pytorch_lighting` style `LightingModule`. Please refer to `lighting.py` for more details.

#### Train from scratch on ImageNet-1k
Train a supernet on the ImageNet from scratch. 
```
python scratch.py --gpus 4 --datapath path/to/imagenet --batch_size 256 --warmup 20 --epochs 500 --learning_rate 0.001
```

Note: When training from scratch on ImageNet-1K with 4 Tesla V100s, it consumes about 1 day per 100 epochs. 

#### Fine-tune on CIFAR10
With a ImageNet-1k pretrained supernet, we can fine-tuning it to CIFAR10. The only difference is that we load the pre-trained weights in the `on_fit_start` of evaluator.
```
python finetune.py --gpus 4 --weights weights/supernet-tiny.pth --datapath path/to/cifar10 --epochs 100 --warmup 5 --learning_rate 0.0005
```

#### Training Detail
Key Hyper-parameter:
- batch size: 128
- criterion: SoftTargetCrossEntropy
- optimizer: optim.AdamW
- weight decay: 0.05
- learning rate: 0.001
- scheduler：LinearWarmupCosineAnnealingLR
- warmup: 20
- epochs: 400

- label smoothing: 0.1
- mixup: 0.8
- cutmix: 1.0
- mixup_prob: 1.0
- mixup_switch_prob: 0.5

### Experiment
Please refer to [nnictl Commands](https://nni.readthedocs.io/zh/stable/reference/nnictl.html) for more detail.

```bash
# view process after experiment done.
nnictl view [-h] [--port PORT] [--experiment_dir EXPERIMENT_DIR] id
# nnictl view --port 8080 --experiment_dir /root/nni-experiments 7e6l8rqd

#stop view.
nnictl stop [-h] [--port PORT] [--all] [id]

# delete experiment
nnictl experiment delete [-h] [--all] [id]


```
### Tools: convert weights
Convert Autoformer official pretrained weight to nni format.

``` bash
mkdir weights
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-tiny.pth -O weights/official-supernet-tiny.pth
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-small.pth -O weights/official-supernet-small.pth
wget https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/supernet-base.pth -O weights/official-supernet-base.pth

python convert.py --name tiny 
```
