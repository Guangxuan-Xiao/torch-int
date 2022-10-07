from struct import pack
from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(
    name='torch_int',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'torch_int._CUDA',
            [
                'torch_int/kernels/gemm.cu',
            ],
            extra_link_args=['-lcublas_static', '-lcublasLt_static', '-lculibos', '-lcudart_static', '-lrt', '-lpthread', '-ldl'],
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)
