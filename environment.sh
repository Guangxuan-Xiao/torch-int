# CUTLASS
export CUTLASS_PATH="/home/guangxuan/cutlass"
export CUDA_INSTALL_PATH="/home/guangxuan/local/cuda-11.8"
export CPATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPATH
export C_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH

# CMAKE
export PATH="/home/guangxuan/local/cmake-3.24.2/bin:$PATH"
# CUDA
export CUDACXX="/home/guangxuan/local/cuda-11.8/bin/nvcc"
export PATH="/home/guangxuan/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/home/guangxuan/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# include path
export CPATH="/home/guangxuan/local/cuda-11.8/include:$CPATH"
# libstdc++
export LD_LIBRARY_PATH="/home/guangxuan/anaconda3/envs/ellm/lib:$LD_LIBRARY_PATH"
