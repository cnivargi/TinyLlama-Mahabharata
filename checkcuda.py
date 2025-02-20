import bitsandbytes as bnb
import torch

print(f"torch version: {torch.__version__}")
print(f"bitsandbytes version: {bnb.__version__}")
print(f"bitsandbytes CUDA backend: {bnb.cuda_available()}")