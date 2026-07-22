import torch
import time

x = torch.randn(10000, 10000, device="cuda")
y = torch.randn(10000, 10000, device="cuda")

torch.cuda.synchronize()
t0 = time.time()
z = x @ y
torch.cuda.synchronize()

print(f"Time: {time.time() - t0:.3f}s")
print(z.device)