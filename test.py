import torch

I, J = 3, 4


mask_i_j_j = torch.triu(torch.ones((I, J, J), device='cpu', dtype=torch.float32)).bool()

print(mask_i_j_j)