import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

# import nni.retiarii.hub.pytorch as hub

from nni.nas.hub.pytorch import AutoformerSpace

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    top1, top5, total = 0, 0, 0
    with tqdm(dataloader) as t:
        for data, target in t:
            data, target = data.to(device), target.to(device)
            topk = torch.topk(model(data), dim=-1, k=5, largest=True, sorted=True).indices
            correct = topk.eq(target.view(-1, 1).expand_as(topk))
            top1 += correct[:, 0].sum().item()
            top5 += correct[:, :5].sum().item()
            total += target.size(0)
            t.set_postfix({"Top1": f"{top1/total:.2%}", "Top5": f"{top5/total:.2%}"})
    return top1/total, top5/total

def main(args):
    model_space = AutoformerSpace(
        search_embed_dim = (192, 216, 240),
        search_mlp_ratio = (3.0, 3.5, 4.0),
        search_num_heads = (3, 4),
        search_depth = (14, 13, 12),
        qkv_bias = True, 
        drop_rate = 0.0, 
        drop_path_rate = 0.1, 
        global_pool = True, 
        num_classes = args.num_classes
    )
    if os.path.exists(args.weights) and os.path.isfile(args.weights):
        model = model_space.load_searched_model(f"autoformer-{args.size}", pretrained=False, download=False)
        model.load_state_dict(torch.load(args.weights))
    else:
        model = model_space.load_searched_model(f"autoformer-{args.size}", pretrained=True, download=True)

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
    parser.add_argument("--weights", type=str, default="path/to/subnet.pth")
    parser.add_argument("--size", choices=["tiny", "small", "base"], type=str, default="tiny")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1000)

    args = parser.parse_args()

    # seed all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)

