import argparse
import os
import random

import numpy as np
import torch

from config import _C
from data_loader import dataset
from learned_simulator import Net
from trainer import Trainer


def arg_parse():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--init', type=str, help='init model from', default='')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='WaterDropSample')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    return parser.parse_args()


def main():
    # ---- setup training environment
    args = arg_parse()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

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

    # ---- setup model
    model = Net()
    model.to(device)

    # ---- setup optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=_C.SOLVER.BASE_LR,
        weight_decay=_C.SOLVER.WEIGHT_DECAY,
    )

    # ---- if resume experiments, use --init ${model_name}
    if args.init:
        print(f'loading pretrained model from {args.init}')
        cp = torch.load(args.init)
        model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    train_set = dataset(data_dir=os.path.join('.data', _C.TRAIN_DIR), phase='train')
    val_set = dataset(data_dir=os.path.join('.data', _C.VAL_DIR), phase='val')
    kwargs = {'pin_memory': False, 'num_workers': 4}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=_C.SOLVER.BATCH_SIZE, shuffle=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=_C.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_loader)} / test {len(val_loader)}')

    # ---- setup trainer
    kwargs = {'device': device,
              'model': model,
              'optim': optim,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'max_iters': _C.SOLVER.MAX_ITERS,
              'dataset': args.dataset}
    trainer = Trainer(**kwargs)

    trainer.train()


if __name__ == '__main__':
    main()