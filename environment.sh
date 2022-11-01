# CUTLASS
export CUTLASS_PATH=~/cutlass
export CUDA_INSTALL_PATH=~/local/cuda-11.8
export CPATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPATH
export C_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH

# CMAKE
export PATH="~/local/cmake-3.24.2/bin:$PATH"
# CUDA
export CUDACXX=~/local/cuda-11.8/bin/nvcc
export PATH="~/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="~/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# include path
export CPATH="~/local/cuda-11.8/include:$CPATH"
# libstdc++
export LD_LIBRARY_PATH=~/anaconda3/envs/ellm/lib:$LD_LIBRARY_PATH
