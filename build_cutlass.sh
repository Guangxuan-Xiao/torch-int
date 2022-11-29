export CUDACXX=/usr/local/cuda/bin/nvcc
cd submodules/cutlass
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j 16
