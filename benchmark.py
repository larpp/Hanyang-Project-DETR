# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Benchmark inference speed of Deformable DETR.
"""
import time
import torch

from argparse_utils import parse_args
from data import make_dataset
from model import Detr
from misc import nested_tensor_from_tensor_list


@torch.no_grad()
def measure_average_inference_time(model, inputs, mask, num_iters=40, warm_iters=5):
    ts = []
    for iter_ in range(num_iters):
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(inputs, mask)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
          ts.append(t)
    print(ts)
    return sum(ts) / len(ts)


def benchmark():
    args = parse_args()
    assert args.warm_iters < args.num_iters and args.num_iters > 0 and args.warm_iters >= 0
    dataset = make_dataset(args.test_directory)
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train=False)
    model.cuda()
    model.eval()
    inputs, mask = nested_tensor_from_tensor_list([dataset.__getitem__(0)[0].cuda() for _ in range(args.batch_size_test)])
    t = measure_average_inference_time(model, inputs, mask, args.num_iters, args.warm_iters)
    return 1.0 / t * args.batch_size_test


if __name__ == '__main__':
    fps = benchmark()
    print(f'Inference Speed: {fps:.1f} FPS')
