import numpy as np
import torch
# def f_1(output):
#   d = output.shape[1]
#   d = torch.from_numpy(np.array(d, dtype='float32'))
#   return (1-torch.exp(-torch.norm(output-1/torch.sqrt(d), p=2)**2))

# def f_2(output):
#   d = output.shape[1]
#   d = torch.from_numpy(np.array(d, dtype='float32'))
#   return (1-torch.exp(-torch.norm(output+1/torch.sqrt(d), p=2)**2))
def f_1(output):
    v = torch.from_numpy(np.array(1/4, dtype='float32'))
    s = torch.from_numpy(np.array(9/2, dtype='float32'))
    return output[0][0]**2 + v*(output[0][1]-s)**2

def f_2(output):
    v = torch.from_numpy(np.array(1/4, dtype='float32'))
    s = torch.from_numpy(np.array(9/2, dtype='float32'))
    return output[0][1]**2 + v*(output[0][0]-s)**2
# def f_1(output):
#     return output[0][0]

# def f_2(output):
#     d = output.shape[1]
#     g = 1 + (9 / (d - 1)) * torch.sum(output[0][1:])
#     return g * (1 - (output[0][0] / g) ** 2)