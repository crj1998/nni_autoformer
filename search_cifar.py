import os, random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


import nni
from nni.nas.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from nni.nas.evaluator.functional import FunctionalEvaluator
from nni.nas.hub.pytorch import AutoformerSpace
from nni.nas.strategy import RandomOneShot, Random, RegularizedEvolution

from nn_meter import load_latency_predictor

from model import builder as model_builder


# @nni.trace
class LatencyFilter:
    def __init__(self, threshold, predictor, predictor_version=None):
        self.predictors = load_latency_predictor(predictor, predictor_version)
        self.threshold = threshold

    def __call__(self, ir_model):
        latency = self.predictors.predict(ir_model, 'nni-ir')
        return latency < self.threshold


@torch.no_grad()
def evaluate_acc(class_cls, model_space, args):
    model_space = deepcopy(model_space)
    # define one-shot strategy
    strategy = RandomOneShot(mutation_hooks=model_space.get_extra_mutation_hooks())
    # attach base model to strategy
    strategy.attach_model(model_space)
    # load pretrained supernet state dict
    model_space.load_state_dict(torch.load(args.weights))

    # get the arch dict of the current sub-model
    arch = nni.get_current_parameter()['mutation_summary']
    # slice supernet params to subnet
    state_dict = strategy.sub_state_dict(arch)
    model = class_cls()
    # load subnet state dict
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset = CIFAR10(args.datapath, download=False, train=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    total, correct = 0, 0
    interval = len(dataloader)//16
    for it, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        total += targets.size(0)
        correct += (logits.argmax(dim=-1)==targets).sum().item()
        acc = correct / total
        if it % interval == 0:
            nni.report_intermediate_result(acc)

    nni.report_final_result(acc)

@nni.trace
def builder(name: str, num_classes: int = 1000):
    return model_builder(name, num_classes)

def main(args):
    model_space = model_builder(name=args.name, num_classes=args.num_classes)

    model_filter = LatencyFilter(threshold=args.latency_threshold, predictor=args.latency_filter) if args.latency_filter else None
    # latency_filter = None
    evaluator = FunctionalEvaluator(evaluate_acc, model_space=model_space, args=args)
    evolution_strategy = Random(model_filter=model_filter)

    exp = RetiariiExperiment(model_space, evaluator, strategy=evolution_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'Random Search(CIFAR10)'
    exp_config.trial_concurrency = args.gpus
    exp_config.trial_gpu_number = 1
    exp_config.max_trial_number = args.max_trial_number
    exp_config.training_service.use_active_gpu = False

    exp.run(exp_config, args.port)

    print('Exported models:')
    for model in exp.export_top_models(formatter='dict'):
        print(model)

if __name__ == "__main__":
    import argparse

    assert torch.cuda.is_available(), "Only work in plaform with CUDA enable."

    parser = argparse.ArgumentParser("AutoFormer Evolutional Search")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--gpus", type=int, default=6)
    parser.add_argument("--datapath", type=str, default="/root/rjchen/data/ImageNet/train")
    parser.add_argument("--weights", type=str, default="./weights/supernet-tiny.pth")
    parser.add_argument("--name", choices=["tiny", "small", "base"], type=str, default="tiny", help="Autoformer size")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max-trial-number", type=int, default=1000)
    parser.add_argument("--evolution-sample-size", type=int, default=50)
    parser.add_argument("--evolution-population-size", type=int, default=50)
    parser.add_argument("--evolution-cycles", type=int, default=100)
    parser.add_argument("--latency-filter", type=str, default=None, help="Apply latency filter by calling the name of the applied hardware.")
    parser.add_argument("--latency-threshold", type=float, default=100, help='inference latency (ms)')
    args = parser.parse_args()

    args.gpus = min(args.gpus, torch.cuda.device_count())
    # assert args.latency_filter in ["cortexA76cpu_tflite21", "adreno640gpu_tflite21", "adreno630gpu_tflite21", "myriadvpu_openvino2019r2"]

    # seed all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True

    main(args)

"""
python search_cifar.py --weights /root/rjchen/workspace/autoformer/weights/finetuned220829.pth --datapath ../../data
"""