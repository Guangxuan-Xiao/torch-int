import torch
import torch.backends.cudnn as cudnn

def bench_model(model, dummy_input, device='cuda', num_iter=100):
    cudnn.benchmark = True
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)
    with torch.no_grad():
        for i in range(num_iter):
            model(dummy_input)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(num_iter):
        model(dummy_input)
    end.record()
    torch.cuda.synchronize()
    print(
        f"Average inference time: {start.elapsed_time(end) / num_iter:.2f} ms")
    torch.cuda.reset_peak_memory_stats()
    for i in range(num_iter):
        model(dummy_input)
    print(
        f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    return start.elapsed_time(end) / num_iter, torch.cuda.max_memory_allocated() / 1024 / 1024

def bench_func(func, args, num_iter=100):
    cudnn.benchmark = True
    # Warm up
    for i in range(num_iter):
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
