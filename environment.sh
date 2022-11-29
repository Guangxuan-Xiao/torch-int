export TORCH_INT_ROOT=$PWD
export CUTLASS_PATH="$TORCH_INT_ROOT/submodules/cutlass"
export CUDA_PATH="/usr/local/cuda"
export PATH="$CUDA_PATH/bin:$PATH"

# CUDA
export CPATH="$CUDA_PATH/include:$CPATH"
export C_INCLUDE_PATH="$CUDA_PATH/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$CUDA_PATH/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# CUTLASS
export CPATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPATH
export C_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUTLASS_PATH/tools/util/include:$CUTLASS_PATH/include:$CPLUS_INCLUDE_PATH


