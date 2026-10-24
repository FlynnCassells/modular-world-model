import time
import torch

print("PyTorch", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(4096, 4096, device=device)

t = time.time()
for _ in range(20):
    y = x @ x
torch.mps.synchronize() if device.type == "mps" else None
print(f"20 matmuls on {device}: {time.time() - t:.2f}s")