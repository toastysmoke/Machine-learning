import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Print CUDA version
    print(f"CUDA version: {torch.version.cuda}")
    
    # Print GPU details
    print(f"GPU: {torch.cuda.get_device_name(0)}")