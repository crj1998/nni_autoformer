import warnings
warnings.filterwarnings("ignore")

import os, random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


from nni.nas.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.nas.strategy import RandomOneShot
import nni.nas.evaluator.pytorch.lightning as pl

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from model import builder as model_builder
from lighting import Classification

def main(args):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    train_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    test_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    train_set = CIFAR10(args.datapath, download=False, train=True, transform=train_transform)
    valid_set = CIFAR10(args.datapath, download=False, train=False, transform=test_transform)

    mixup = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, 
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
        label_smoothing=args.smoothing, num_classes=args.num_classes
    )

    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    if os.path.exists(args.weights) and os.path.isfile(args.weights):
        state_dict = torch.load(args.weights)
        state_dict["head.weight"] = torch.randn_like(state_dict["head.weight"][:10])
        state_dict["head.bias"] = torch.randn_like(state_dict["head.bias"][:10])
    else:
        state_dict = None
    
    evaluator = Classification(
        weights = state_dict,
        criterion = criterion,
        optimizer = optim.AdamW,
        scheduler = LinearWarmupCosineAnnealingLR,
        mixup = mixup,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        warmup = args.warmup,
        # Need to use `pl.DataLoader` instead of `torch.utils.data.DataLoader` here,
        # or use `nni.trace` to wrap `torch.utils.data.DataLoader`.
        train_dataloaders = pl.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True),
        val_dataloaders = pl.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True),
        # Other keyword arguments passed to pytorch_lightning.Trainer.
        max_epochs = args.epochs,
        gpus = args.gpus,
        accelerator = "gpu",
        strategy = "ddp"
    )

    model_space = model_builder(args.name, args.num_classes)

    strategy = RandomOneShot(mutation_hooks=model_space.get_extra_mutation_hooks())

    exp = RetiariiExperiment(model_space, evaluator, [], strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'CIFAR10 fine-tuning'
    exp_config.execution_engine = 'oneshot'

    exp.run(exp_config, args.port)

    torch.save(model_space.state_dict(), "weights/finetuned220830.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("AutoFormer fine-tune")
    parser.add_argument("--port", type=int, default=6002)
    parser.add_argument("--name", choices=["tiny", "small", "base"], type=str, default="tiny", help="Autoformer size")
    parser.add_argument("--datapath", type=str, default="path/to/cifar10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--weights", type=str, required=True, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--smoothing", type=float, default=0.1, help="label smoothing")

    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')

    args = parser.parse_args()



    # seed all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True

    args.gpus = min(args.gpus, torch.cuda.device_count())

    main(args)

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --weights /root/rjchen/workspace/autoformer/weights/supernet20220822.pth --datapath /root/rjchen/data --epochs 50 --warmup 5 --gpus 4 --learning_rate 0.0005
"""