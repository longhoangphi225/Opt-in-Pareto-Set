import torch
def linear_function(losses,ray):
    return ray[0]*losses[0]+ ray[1]*losses[1]
def quadratic_function(losses,ray):
    return ray[0]*(losses[0]-2)**2 + ray[1]*(losses[1]-2)**2
def wd_function(losses,ray):
    return (1/ray[0])*(losses[0]-ray[0])**2 + (1/ray[1])*(losses[1]-ray[1])**2
def utility_function(losses,ray):
    return ((losses[0]+4)**ray[0])*((losses[1]+4)**ray[1])
    #return ray[0]*torch.log(losses[0]+1) + ray[1]*torch.log(losses[1]+1)
def complex_cos_F(losses,ray):
    energy = - torch.cos(0.5 * 3.14159 * (losses[0] - ray[0])) * ((1. + torch.cos(3.14159 * (losses[1] - ray[1]))) ** 2)
    return energy