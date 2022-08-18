import os, random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


import nni
from nni.nas.hub.pytorch import AutoformerSpace
from nni.nas.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
from nni.nas.evaluator.functional import FunctionalEvaluator
from nni.nas.strategy import RandomOneShot, Random, RegularizedEvolution

from nn_meter import load_latency_predictor

# @nni.trace
class LatencyFilter:
    def __init__(self, threshold, predictor, predictor_version=None):
        self.predictors = load_latency_predictor(predictor, predictor_version)
        self.threshold = threshold

    def __call__(self, ir_model):
        latency = self.predictors.predict(ir_model, 'nni-ir')
        return latency < self.threshold


@torch.no_grad()
def evaluate_acc(class_cls, base_model, args):
    model = class_cls()
    # deepcopy model
    base_model = deepcopy(base_model)
    # define one-shot strategy
    strategy = RandomOneShot(mutation_hooks=base_model.get_extra_mutation_hooks())
    # attach base model to strategy
    strategy.attach_model(base_model)
    # load pretrained supernet state dict
    base_model.load_state_dict(torch.load(args.weights))
    # slice supernet params to subnet
    state_dict = strategy.sub_state_dict(model)
    # load subnet state dict
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    dataset = ImageFolder(args.datapath, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    total, correct = 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        total += targets.size(0)
        correct += (logits.argmax(dim=-1)==targets).sum().item()
        acc = correct / total
        nni.report_intermediate_result(acc)

    nni.report_final_result(acc)


def main(args):

    base_model = nni.trace(AutoformerSpace)(
        search_embed_dim = (192, 216, 240),
        search_mlp_ratio = (3.0, 3.5, 4.0),
        search_num_heads = (3, 4),
        search_depth = (12, 13, 14),
        qkv_bias = True, 
        drop_rate = 0.0, 
        drop_path_rate = 0.1, 
        global_pool = True,
        num_classes = 1000
    )
    
    model_filter = nni.trace(LatencyFilter)(threshold=args.latency_threshold, predictor=args.latency_filter) if args.latency_filter else None
    # latency_filter = None
    evaluator = FunctionalEvaluator(evaluate_acc, base_model=base_model, args=args)
    evolution_strategy = Random(model_filter=model_filter)
    evolution_strategy = RegularizedEvolution(
        sample_size=args.evolution_sample_size, 
        population_size=args.evolution_population_size, 
        cycles=args.max_trial_number,
        model_filter=model_filter,
    )
    
    exp = RetiariiExperiment(base_model, evaluator, strategy=evolution_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'Random Evolutionary Search'
    exp_config.trial_concurrency = 1
    exp_config.trial_gpu_number = 1
    exp_config.max_trial_number = args.max_trial_number
    exp_config.training_service.use_active_gpu = False

    exp.run(exp_config, args.port)

    print('Exported models:')
    for i, model in enumerate(exp.export_top_models(formatter='dict')):
        print(model)

if __name__ == "__main__":
    """
    python randomsearch.py
    """
    import argparse
    parser = argparse.ArgumentParser("AutoFormer Evolutional Search")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--datapath", type=str, default="/root/rjchen/data/ImageNet/train")
    parser.add_argument("--weights", type=str, default="./weights/supernet-tiny.pth")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max-trial-number", type=int, default=3)
    parser.add_argument("--evolution-sample-size", type=int, default=10)
    parser.add_argument("--evolution-population-size", type=int, default=50)
    parser.add_argument("--evolution-cycles", type=int, default=300)

    args = parser.parse_args()

    args.gpus = min(args.gpus, torch.cuda.device_count())
    # assert args.latency_filter in ["cortexA76cpu_tflite21", "adreno640gpu_tflite21", "adreno630gpu_tflite21", "myriadvpu_openvino2019r2"]

    # seed all
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True

    main(args)