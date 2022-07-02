import numpy as np
import torch
from problems import f_1, f_2
def concave_fun_eval(x):
    return np.stack([f_1(x).item(),f_2(x).item()])
def concave_fun_eval1(x):
    return torch.Tensor([[9*x/(2*x-8),9/(2-8*x)]])
def concave_fun_eval2(x):
    return torch.Tensor([[x,2-x]])
def concave_fun_eval3(x):
    return torch.Tensor([[9/(2-8*x),9*x/(2*x-8)]])
def create_pf():
    ps = np.linspace(-1/np.sqrt(2),1/np.sqrt(2))
    pf = []
    for x1 in ps:
        x = torch.Tensor([[x1,x1]])
        f = concave_fun_eval(x)
        pf.append(f)   
    pf = np.array(pf)
    return pf
def create_pf1():
    ps1 = np.linspace(-1 / 2, 0, num=500)
    ps3 = np.linspace(-1 / 2, 0, num=500)
    ps2 = np.linspace(1/2, 3/2, num=500)
    pf = []
    for x1 in ps1:
        x = concave_fun_eval1(x1)
        f = concave_fun_eval(x)
        pf.append(f)
    for x2 in ps2:
        x = concave_fun_eval2(x2)
        f = concave_fun_eval(x)
        pf.append(f)    
    for x3 in ps3:
        x = concave_fun_eval3(x3)
        f = concave_fun_eval(x)
        pf.append(f)     
    pf = np.array(pf)
    return pf
def create_pf2():
    f1 = np.linspace(0.0, 1.0, 500)
    f2 = 1 - f1 ** 2
    pf = []
    tmp = np.stack([f1,f2])
    pf.append(tmp) 
    pf = np.array(pf)[0].T
    return pf