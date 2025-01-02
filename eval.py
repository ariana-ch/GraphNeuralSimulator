import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import _C as C
from data_loader import dataset
from evaluator import PredEvaluator
from learned_simulator import Net


def arg_parse():
    parser = argparse.ArgumentParser(description='Eval parameters')
    parser.add_argument('--ckpt', type=str, help='', default=None)
    parser.add_argument('--dataset', type=str, default='WaterDropSample')
    return parser.parse_args()


def main():
    args = arg_parse()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        device = torch.device('cuda')
    else:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # --- setup config files
    model_name = args.dataset
    iter_name = args.ckpt.split('/')[-1].split('.')[0]
    output_dir = os.path.join('animations', model_name, iter_name)

    # --- setup data loader
    test_set = dataset(data_dir=os.path.join('.data', args.dataset, 'test'), phase='val')
    data_loader = DataLoader(test_set, batch_size=C.SOLVER.BATCH_SIZE, num_workers=0)

    # ---- setup model
    model = Net()
    model.to(device)

    cp = torch.load(args.ckpt, map_location=f'{device.type}:0')
    model.load_state_dict(cp['model'])
    tester = PredEvaluator(
        device=torch.device(device),
        data_loader=data_loader,
        model=model,
        output_dir=output_dir,
    )
    tester.test()


if __name__ == '__main__':
    main()