from struct import pack
from numpy import source
from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(
    name='torch_int',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='torch_int._CUDA',
            sources=[
                'torch_int/kernels/bmm.cu',
                'torch_int/kernels/linear.cu',
                'torch_int/kernels/fused.cu',
                'torch_int/kernels/bindings.cpp',
                'torch_int/kernels/gemm_cublas.cu',
                'torch_int/kernels/gemm_cutlass.cu',
                'torch_int/kernels/quantization.cu',
            ],
            include_dirs=['torch_int/kernels/include'],
            extra_link_args=['-lcublas', '-lcublasLt',
                             '-lculibos', '-lcudart', '-lrt', '-lpthread', '-ldl'],
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)
