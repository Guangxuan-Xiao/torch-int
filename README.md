# torch-int
Integer operators for PyTorch

## Dependencies
- PyTorch
- Cutlass
- Cublas
- Nvidia-Toolkit

## Installation
```bash
conda create -n int python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

python setup.py install
```