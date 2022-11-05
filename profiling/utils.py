import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
import torch.backends.cudnn as cudnn
from tqdm import trange


@torch.no_grad()
def profile_model(model, inputs, export_path, device='cuda', num_iter=100):
    cudnn.benchmark = True
    model.eval()
    model.to(device)
    inputs = tuple(input.to(device) for input in inputs)
    print('Warming up...')
    for _ in trange(10):
        model(*inputs)
    print('Profiling...')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            for _ in trange(num_iter):
                model(*inputs)
    os.makedirs(export_path, exist_ok=True)
    print('Exporting profile to', export_path)
    profile_text_path = os.path.join(export_path, 'profile.txt')
    with open(profile_text_path, 'w') as f:
        print("CPU Time total:", file=f)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20),
              file=f)
        print("CUDA Time total:", file=f)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
              file=f)
        print("CPU Memory:", file=f)
        print(prof.key_averages().table(
            sort_by="self_cpu_memory_usage", row_limit=20), file=f)
        print("CUDA Memory:", file=f)
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=20), file=f)
    time_stacks_path = os.path.join(export_path, 'cuda_time.stacks')
    prof.export_stacks(time_stacks_path, "self_cuda_time_total")
    # Generate a flame graph
    flame_graph_path = os.path.join(export_path, 'cuda_time_flame.svg')
    os.system(
        f'~/repos/FlameGraph/flamegraph.pl --title "CUDA Time" --countname "us." {time_stacks_path} > {flame_graph_path}')
