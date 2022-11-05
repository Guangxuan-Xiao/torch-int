
# CUTLASS
export CUTLASS_PATH="$HOME/cutlass"
export CUDA_INSTALL_PATH="$HOME/local/cuda-11.8"
export CPATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPATH
export C_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH

# CMAKE
export PATH="$HOME/local/cmake-3.24.2/bin:$PATH"
# CUDA
export CUDACXX="$HOME/local/cuda-11.8/bin/nvcc"
export PATH="$HOME/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# include path
export CPATH="$HOME/local/cuda-11.8/include:$CPATH"
# libstdc++
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/ellm/lib:$LD_LIBRARY_PATH"
