import torch
from torch_int.nn import Int8Linear
from utils import benchmark


if __name__ == '__main__':
    SEQ_LEN = 128
    HIDDEN_SIZE = 32768  # 32K
    model_int8 = Int8Linear(HIDDEN_SIZE, HIDDEN_SIZE)  # 32K x 32K = 1G
    dummy_input = torch.randn(SEQ_LEN, HIDDEN_SIZE).half()
    print("Int8Linear:")
    t_int8, m_int8 = benchmark(model_int8, dummy_input.to(torch.int8))
    model_fp16 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE).half()
    print("FP16Linear:")
    t_fp16, m_fp16 = benchmark(model_fp16, dummy_input)
    print(f"Int8Linear is {t_fp16 / t_int8:.2f}x faster than FP16Linear")
    print(
        f"Int8Linear uses {m_fp16 / m_int8:.2f}x less memory than FP16Linear")

    # Results on V100:
    # Int8Linear:
    # Average inference time: 5.63 ms
    # Peak memory usage: 1052.06 MB
    # FP16Linear:
    # Average inference time: 6.90 ms
    # Peak memory usage: 3088.12 MB
    # Int8Linear is 1.23x faster than FP16Linear
    # Int8Linear uses 2.94x less memory than FP16Linear
