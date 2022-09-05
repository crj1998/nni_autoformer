import os, random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as T


import nni
from nni.nas.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from nni.nas.evaluator.functional import FunctionalEvaluator
from nni.nas.strategy import RandomOneShot, Random, RegularizedEvolution
from nni.nas.hub.pytorch.utils.pretrained import load_pretrained_weight

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
    if os.path.exists(args.weights) and os.path.isfile(args.weights):
        # define one-shot strategy
        strategy = RandomOneShot(mutation_hooks=model_space.get_extra_mutation_hooks())
        # attach base model to strategy
        strategy.attach_model(model_space)
        # load pretrained supernet state dict
        super_state_dict = torch.load(args.weights)
        strategy.model.load_state_dict(super_state_dict)
    else:
        model_space.load_strategy_checkpoint(f'random-one-shot-{args.name}')
    # get the arch dict of the current sub-model
    arch = nni.get_current_parameter()['mutation_summary']
    # slice supernet params to subnet
    state_dict = strategy.sub_state_dict(arch)
    model = class_cls()
    # load subnet state dict
    model.load_state_dict(state_dict)
    model.eval().cuda()

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    if args.dataset.lower() == "cifar10":
        dataset = CIFAR10(args.datapath, download=False, train=False, transform=transform)
    else:
        dataset = ImageFolder(args.datapath, transform=transform)
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


def main(args):
    model_space = model_builder(name=args.name, num_classes=args.num_classes, nni_traced=True)

    model_filter = LatencyFilter(threshold=args.latency_threshold, predictor=args.latency_filter) if args.latency_filter else None
    evaluator = FunctionalEvaluator(evaluate_acc, model_space=model_space, args=args)
    if args.evolution:
        evolution_strategy = RegularizedEvolution(
            sample_size=args.evolution_sample_size,
            population_size=args.evolution_population_size,
            cycles=args.max_trial_number,
            model_filter=model_filter,
        )
    else:
        evolution_strategy = Random(model_filter=model_filter)


    exp = RetiariiExperiment(model_space, evaluator, strategy=evolution_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = f'Random Search({args.dataset.upper()})'
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
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "imagenet"])
    parser.add_argument("--datapath", type=str, default="/root/rjchen/data/ImageNet/train")
    parser.add_argument("--weights", type=str, default="./weights/supernet-tiny.pth")
    parser.add_argument("--name", choices=["tiny", "small", "base"], type=str, default="tiny", help="Autoformer size")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max-trial-number", type=int, default=512)
    parser.add_argument("--evolution", action="store_true")
    parser.add_argument("--evolution-sample-size", type=int, default=50)
    parser.add_argument("--evolution-population-size", type=int, default=50)
    parser.add_argument("--evolution-cycles", type=int, default=100)
    parser.add_argument("--latency-filter", type=str, default=None, help="Apply latency filter by calling the name of the applied hardware.")
    parser.add_argument("--latency-threshold", type=float, default=100, help='inference latency (ms)')
    args = parser.parse_args()

    args.gpus = min(args.gpus, torch.cuda.device_count())
    if args.dataset.lower() == "cifar10":
        args.num_classes == 10
    # assert args.latency_filter in ["cortexA76cpu_tflite21", "adreno640gpu_tflite21", "adreno630gpu_tflite21", "myriadvpu_openvino2019r2"]

    # seed all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True

    main(args)

"""
python search.py --weights weights/finetuned220829.pth --dataset cifar10 --num_classes 10 --datapath ../../data
python search.py --weights weights/supernet20220822.pth --dataset imagenet --num_classes 1000 --datapath ../../data/imagenet/val
python search.py --name tiny --dataset imagenet --num_classes 1000 --datapath ../../data/imagenet/val
python search.py --name tiny --weights weights/supernet-tiny.pth --dataset imagenet --num_classes 1000 --datapath ../../data/imagenet/val
"""