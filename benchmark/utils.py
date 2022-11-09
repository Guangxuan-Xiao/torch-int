import torch
import torch.backends.cudnn as cudnn
from typing import Tuple
from tqdm import trange

@torch.no_grad()
def bench_generation(model, inputs: Tuple, device='cuda', num_iter=1000):
    cudnn.benchmark = True
    model.eval()
    model.to(device)
    inputs = tuple(input.to(device) for input in inputs)
    for i in range(10):
        model(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in trange(num_iter):
        model(*inputs)
    end.record()
    torch.cuda.synchronize()
    print(
        f"Average inference time: {start.elapsed_time(end) / num_iter:.2f} ms")
    torch.cuda.reset_peak_memory_stats()
    for i in range(10):
        model(*inputs)
    print(
        f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    return start.elapsed_time(end) / num_iter, torch.cuda.max_memory_allocated() / 1024 / 1024


@torch.no_grad()
def bench_model(model, inputs: Tuple, device='cuda', num_iter=1000):
    cudnn.benchmark = True
    model.eval()
    model.to(device)
    inputs = tuple(input.to(device) for input in inputs)
    for i in range(10):
        model(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in trange(num_iter):
        model(*inputs)
    end.record()
    torch.cuda.synchronize()
    print(
        f"Average inference time: {start.elapsed_time(end) / num_iter:.2f} ms")
    torch.cuda.reset_peak_memory_stats()
    for i in range(10):
        model(*inputs)
    print(
        f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    return start.elapsed_time(end) / num_iter, torch.cuda.max_memory_allocated() / 1024 / 1024


def bench_func(func, args, num_iter=1000):
    cudnn.benchmark = True
    # Warm up
    for i in range(100):
        func(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(num_iter):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    print(
        f"Average inference time: {start.elapsed_time(end) / num_iter:.2f} ms")
    torch.cuda.reset_peak_memory_stats()
    for i in range(num_iter):
        func(*args)
    print(
        f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    return start.elapsed_time(end) / num_iter, torch.cuda.max_memory_allocated() / 1024 / 1024


def bench_func_latency(func, args, num_iter=1000):
    cudnn.benchmark = True
    # Warm up
    for i in range(100):
        func(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(num_iter):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    print(
        f"Average inference time: {start.elapsed_time(end) / num_iter} ms")
    return start.elapsed_time(end) / num_iter
