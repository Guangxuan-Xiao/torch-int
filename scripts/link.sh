g++ -pthread -B /home/xgx/anaconda3/compiler_compat -shared -L/home/xgx/anaconda3/lib -Wl,-rpath,/home/xgx/anaconda3/lib -Wl,-rpath-link,/home/xgx/anaconda3/lib \
    build/temp.linux-x86_64-cpython-39/torch_int/kernels/bindings.o \
    build/temp.linux-x86_64-cpython-39/torch_int/kernels/bmm.o \
    build/temp.linux-x86_64-cpython-39/torch_int/kernels/fused.o \
    build/temp.linux-x86_64-cpython-39/torch_int/kernels/gemm_cublas.o \
    build/temp.linux-x86_64-cpython-39/torch_int/kernels/gemm_cutlass.o \
    build/temp.linux-x86_64-cpython-39/torch_int/kernels/linear.o \
    build/temp.linux-x86_64-cpython-39/torch_int/kernels/quantization.o \
    -L/home/xgx/anaconda3/lib/python3.9/site-packages/torch/lib \
    -L/usr/local/cuda/lib64 \
    -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lcudart_static -lc10_cuda -ltorch_cuda_cu -ltorch_cuda_cpp -o \
    build/lib.linux-x86_64-cpython-39/torch_int/_CUDA.cpython-39-x86_64-linux-gnu.so \
    -lcublas_static -lcublasLt_static -lculibos -lcudart -lrt -lpthread -ldl -lcudart_static

ln -s build/lib.linux-x86_64-cpython-39/torch_int/_CUDA.cpython-39-x86_64-linux-gnu.so torch_int/_CUDA.cpython-39-x86_64-linux-gnu.so