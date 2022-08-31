import os
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from nni.nas.strategy import RandomOneShot
from nni.nas import fixed_arch

from model import builder as model_builder

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    top1, top5, total = 0, 0, 0
    with tqdm(dataloader, total=len(dataloader), desc="Valid", ncols=100) as t:
        for inputs, targets in t:
            inputs, targets = inputs.to(device), targets.to(device)
            topk = torch.topk(model(inputs), dim=-1, k=5, largest=True, sorted=True).indices
            correct = topk.eq(targets.view(-1, 1).expand_as(topk))
            top1 += correct[:, 0].sum().item()
            top5 += correct[:, :5].sum().item()
            total += targets.size(0)
            t.set_postfix({"Top1": f"{top1/total:.2%}", "Top5": f"{top5/total:.2%}"})
    return top1/total, top5/total


def main(args):
    model_space = model_builder(args.name, args.num_classes)

    if isinstance(args.arch, dict):
        strategy = RandomOneShot(mutation_hooks=model_space.get_extra_mutation_hooks())
        strategy.attach_model(model_space)

        strategy.model.load_state_dict(torch.load(args.weights))
        args.arch = strategy.model.resample(args.arch)

        with fixed_arch(args.arch):
            model = model_builder(args.name, args.num_classes)
            state_dict = strategy.sub_state_dict(args.arch)
            model.load_state_dict(state_dict)
    else:
        if os.path.exists(args.weights) and os.path.isfile(args.weights):
            model = model_space.load_searched_model(f"autoformer-{args.name}", pretrained=False, download=False)
            model.load_state_dict(torch.load(args.weights))
        else:
            model = model_space.load_searched_model(f"autoformer-{args.name}", pretrained=True, download=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset = ImageFolder(args.datapath, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    top1_acc, top5_acc = validate(model, dataloader, device)
    print(f"ImageNet acc@top1: {top1_acc:.2%}, acc@top5: {top5_acc:.2%}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("AutoFormer Subnet Evaluate")
    parser.add_argument("--datapath", type=str, default="path/to/imagenet/val")
    parser.add_argument("--weights", type=str, default="path/to/supernet.pth")
    parser.add_argument("--name", choices=["tiny", "small", "base"], type=str, default="tiny", help="Autoformer size")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1000)

    args = parser.parse_args()
    args.arch = None
    # args.arch = {'embed_dim': 240, 'depth': 12, 'num_head_0': 3, 'mlp_ratio_0': 4.0, 'num_head_1': 4, 'mlp_ratio_1': 3.0, 'num_head_2': 3, 'mlp_ratio_2': 3.0, 'num_head_3': 3, 'mlp_ratio_3': 4.0, 'num_head_4': 3, 'mlp_ratio_4': 4.0, 'num_head_5': 4, 'mlp_ratio_5': 3.0, 'num_head_6': 3, 'mlp_ratio_6': 3.0, 'num_head_7': 3, 'mlp_ratio_7': 3.0, 'num_head_8': 3, 'mlp_ratio_8': 4.0, 'num_head_9': 3, 'mlp_ratio_9': 4.0, 'num_head_10': 4, 'mlp_ratio_10': 3.0, 'num_head_11': 4, 'mlp_ratio_11': 4.0, 'num_head_12': 4, 'mlp_ratio_12': 3.0, 'num_head_13': 3, 'mlp_ratio_13': 4.0}
    # args.arch = {'embed_dim': 192, 'depth': 13}
    # seed all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)

"""
python evaluate.py --datapath ../../data/imagenet/val
"""