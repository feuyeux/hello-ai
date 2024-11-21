import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Use GPU for computations")
else:
    device = torch.device("cpu")
    print("Fallback to CPU")

# Create tensors on the chosen device (GPU or CPU)
a = torch.randn(2, 3, device=device)
b = torch.randn(3, 4, device=device)

# Perform matrix multiplication on the chosen device
c = torch.mm(a, b)

# Print the result (tensor on the same device)
print(c)
