from time import time
import torch
from torch import nn
import logging
import time
import numpy as np
import random
from models import Hypernetwork
from methods import LinearScalarizationSolver
import numpy as np
import torch
from matplotlib import pyplot as plt
from constrain import PNG_solver
from problems import f_1, f_2
from create_pareto_front import create_pf, create_pf1, create_pf2
from extra_criterion_functions import linear_function, quadratic_function, wd_function, utility_function, complex_cos_F
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
color_list = ['#28B463', '#326CAE', '#FFC300', '#FF5733']
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
                'size': 18,
               }
def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_logger()

device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # def f_1(output):
# #   d = output.shape[1]
# #   d = torch.from_numpy(np.array(d, dtype='float32'))
# #   return (1-torch.exp(-torch.norm(output-1/torch.sqrt(d), p=2)**2))

# # def f_2(output):
# #   d = output.shape[1]
# #   d = torch.from_numpy(np.array(d, dtype='float32'))
# #   return (1-torch.exp(-torch.norm(output+1/torch.sqrt(d), p=2)**2))
# def f_1(output):
#     v = torch.from_numpy(np.array(1/4, dtype='float32'))
#     s = torch.from_numpy(np.array(9/2, dtype='float32'))
#     return output[0][0]**2 + v*(output[0][1]-s)**2

# def f_2(output):
#     v = torch.from_numpy(np.array(1/4, dtype='float32'))
#     s = torch.from_numpy(np.array(9/2, dtype='float32'))
#     return output[0][1]**2 + v*(output[0][0]-s)**2
# # def f_1(output):
# #     return output[0][0]

# # def f_2(output):
# #     d = output.shape[1]
# #     g = 1 + (9 / (d - 1)) * torch.sum(output[0][1:])
# #     return g * (1 - (output[0][0] / g) ** 2)

# def linear_function(losses,ray):
#     return ray[0]*losses[0]+ ray[1]*losses[1]
# def quadratic_function(losses,ray):
#     return ray[0]*(losses[0]-2)**2 + ray[1]*(losses[1]-2)**2
# def wd_function(losses,ray):
#     return (1/ray[0])*(losses[0]-ray[0])**2 + (1/ray[1])*(losses[1]-ray[1])**2
# def utility_function(losses,ray):
#     return ((losses[0]+4)**ray[0])*((losses[1]+4)**ray[1])
#     #return ray[0]*torch.log(losses[0]+1) + ray[1]*torch.log(losses[1]+1)
# def complex_cos_F(losses,ray):
#     energy = - torch.cos(0.5 * 3.14159 * (losses[0] - ray[0])) * ((1. + torch.cos(3.14159 * (losses[1] - ray[1]))) ** 2)
#     return energy
# def concave_fun_eval(x):
#     return np.stack([f_1(x).item(),f_2(x).item()])
# def concave_fun_eval1(x):
#     return torch.Tensor([[9*x/(2*x-8),9/(2-8*x)]])
# def concave_fun_eval2(x):
#     return torch.Tensor([[x,2-x]])
# def concave_fun_eval3(x):
#     return torch.Tensor([[9/(2-8*x),9*x/(2*x-8)]])
    
# def create_pf():
#     ps = np.linspace(-1/np.sqrt(2),1/np.sqrt(2))
#     pf = []
#     for x1 in ps:
#         x = torch.Tensor([[x1,x1]])
#         f = concave_fun_eval(x)
#         pf.append(f)   
#     pf = np.array(pf)
#     return pf
# def create_pf1():
#     ps1 = np.linspace(-1 / 2, 0, num=500)
#     ps3 = np.linspace(-1 / 2, 0, num=500)
#     ps2 = np.linspace(1/2, 3/2, num=500)
#     pf = []
#     for x1 in ps1:
#         x = concave_fun_eval1(x1)
#         f = concave_fun_eval(x)
#         pf.append(f)
#     for x2 in ps2:
#         x = concave_fun_eval2(x2)
#         f = concave_fun_eval(x)
#         pf.append(f)    
#     for x3 in ps3:
#         x = concave_fun_eval3(x3)
#         f = concave_fun_eval(x)
#         pf.append(f)     
#     pf = np.array(pf)
#     return pf
# def create_pf2():
#     f1 = np.linspace(0.0, 1.0, 500)
#     f2 = 1 - f1 ** 2
#     pf = []
#     tmp = np.stack([f1,f2])
#     pf.append(tmp) 
#     pf = np.array(pf)[0].T
#     return pf
#pf = create_pf() # pareto for OF04
pf = create_pf1() # pareto for OF01
#pf = create_pf2() # pareto for OF03

def train(device, hidden_dim, lr, wd, epochs, alpha, alpha_r, outdim, step_size, max_iters, criterion, type, threshold,n_tasks):

    e_solver = PNG_solver(max_iters = max_iters, n_dim = outdim, step_size = step_size)
    hnet: nn.Module = Hypernetwork(ray_hidden_dim = hidden_dim, out_dim = outdim, n_tasks = 1)
    logging.info(f"HN size: {count_parameters(hnet)}")
    hnet = hnet.to(device)
    loss1 = f_1
    loss2 = f_2
    solver = LinearScalarizationSolver(n_tasks = n_tasks)
    losses_ = []
    sol = []
    if type == "2 phrases":
        print("Start phrase 1")
        optimizer = torch.optim.Adam(hnet.parameters(), lr = 1e-3, weight_decay = wd)
        for epoch in range(epochs):
            ray = torch.from_numpy(
                np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
            ).to(device)
            hnet.train()
            optimizer.zero_grad()
            output = hnet(ray)
            l1 = loss1(output)
            l2 = loss2(output)
            losses = torch.stack((l1, l2))
            x0 = output.detach().cpu().numpy()[0]
            x0 = torch.tensor(list(x0)).float().unsqueeze(0)
            x0.requires_grad = True
            d = e_solver.optimize(x0, criterion = criterion, context = ray.detach().cpu().numpy().tolist(), alpha = alpha, threshold = threshold, start=-1)
            ray = ray.squeeze(0)
            lda = torch.Tensor([d.detach().cpu().numpy()[0][0][0],d.detach().cpu().numpy()[1][0][0]])
            lda = lda.squeeze(0).to(device)
            loss = solver(losses, lda, list(hnet.parameters())) 
            loss.backward()
            optimizer.step()
        torch.save(hnet,("weight1.pt"))
        hnet = torch.load("weight1.pt")
        print("Start phrase 2")


    optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
    start = time.time()
    for epoch in range(epochs):
        ray = torch.from_numpy(
            np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
        ).to(device)
        hnet.train()
        optimizer.zero_grad()
        output = hnet(ray)
        x0 = output.detach().cpu().numpy()[0]
        x0 = torch.tensor(list(x0)).float().unsqueeze(0)
        x0.requires_grad = True
        d = e_solver.optimize(x0, criterion = criterion, context=ray.detach().cpu().numpy().tolist(), alpha = alpha, threshold = threshold, start=-1)
        l1 = loss1(output)
        l2 = loss2(output)
        losses = torch.stack((l1, l2))
        ray = ray.squeeze(0)
        lda = torch.Tensor([d.detach().cpu().numpy()[0][0][0],d.detach().cpu().numpy()[1][0][0]])
        lda = lda.squeeze(0).to(device)
        loss_L = solver(losses, lda, list(hnet.parameters()))
        # cosin complex, weighted distance, utility, linear, quadratic convex
        if criterion == "cosin complex":
            loss_F = complex_cos_F(losses,ray)
        elif criterion == "utility":
            loss_F = utility_function(losses,ray)
        elif criterion == "linear":
            loss_F = linear_function(losses,ray)
        elif criterion == "quadratic convex":
            loss_F = quadratic_function(losses,ray)
        elif criterion == "weighted distance":
            loss_F = wd_function(losses,ray)
        if type == "2 phrases":
            loss = loss_F
        else:
            loss = loss_F + loss_L
        loss.backward()
        optimizer.step()
        losses_.append(loss.item())
        sol.append(output.cpu().detach().numpy().tolist()[0])
    end = time.time()
    time_training = end-start
    torch.save(hnet,("best_weight.pt"))
    return sol,time_training

def find_target(pf, criterion, context=None):
    # cosin complex, weighted distance, utility, linear, quadratic convex
    if criterion == 'template':
        F = context[1] * (pf[:, 0] - context[0]) ** 2 + context[0] * (pf[:, 1] - context[1]) ** 2
        # F = context[0] * (pf[:, 0] - context[0]) ** 2 + context[1] * (pf[:, 1] - context[1]) ** 2
        return pf[F.argmin(), :]
    elif criterion == 'weighted distance':
        F = (1/context[0])*(pf[:, 0] - context[0]) ** 2 + (1/context[1])*(pf[:, 1] - context[1]) ** 2
        return pf[F.argmin(), :]
    elif criterion == 'cosin complex':
        F = -np.cos(0.5 * 3.14159 * (pf[:, 0] - context[0])) * ((1+np.cos(3.14159 * (pf[:, 1] - context[1]))) ** 2)    
        return pf[F.argmin(), :]
    elif criterion == 'linear':
        F = context[0]*pf[:, 0] + context[1]*pf[:, 1]
        return pf[F.argmin(), :]
    elif criterion == 'quadratic convex':
        F = context[0]*(pf[:, 0]-2)**2 + context[1]*(pf[:, 1]-2)**2
        return pf[F.argmin(), :]
    elif criterion == 'utility':
        #F = context[0]*np.log(pf[:, 0]) + context[1]*np.log(pf[:, 1])
        F = ((pf[:, 0]+4)**context[0])*((pf[:, 1]+4)**context[1])
        return pf[F.argmin(), :]
def predict(time_training,criterion):

    hnet = torch.load("best_weight.pt")
    hnet.eval()
    loss1 = f_1
    loss2 = f_2
    results = []
    contexts = [[0.2, 0.8], [0.4, 0.6],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.6,0.4],[0.1,0.9],[0.9,0.1]]
    fig, ax = plt.subplots()
    targets = []
    # for i in contexts:
    #     # ray = torch.from_numpy(
    #     #         np.array([1e-3*i, np.sqrt(1-(1e-3*i)**2)], dtype='float32')
    #     #     ).to(device)
    #     ray = torch.Tensor(i).to(device)
    #     output = hnet(ray)
    #     l1 = loss1(output)
    #     l2 = loss2(output)
    #     results.append([l1, l2])
    #     #target = find_target(pf, criterion="complex cos", context=i)
    #     target = find_target(pf, criterion="utility", context=i)
    #     targets.append(target)

    for i in range(100):
        tmp = np.random.rand()
        ray = torch.Tensor([tmp,1-tmp]).to(device)
        output = hnet(ray)
        l1 = loss1(output)
        l2 = loss2(output)
        results.append([l1, l2])
        target = find_target(pf, criterion = criterion, context = [tmp,1-tmp])
        targets.append(target)
    targets = np.array(targets)
    ax.scatter(targets[:,0], targets[:,1], s=500,c=color_list[1], marker='*', alpha=1, label='Target', zorder=0)
    results = [[i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy()] for i in results]
    results = np.array(results, dtype='float32')
    ax.set_xlabel(r'$l_1$')
    ax.set_ylabel(r'$l_2$')
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)
    ax.scatter(results[:, 0], results[:, 1],s=100,c=color_list[0],label='HPN-PNGD')
    ax.scatter(pf[:,0],pf[:,1],s=10,c='gray',label='Pareto Front',zorder=0)
    #ax.plot(pf[:,0],pf[:,1],c='gray',label='Pareto Front',zorder=0)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
    plt.legend(prop=font_legend)
    plt.title('Prediction TC02 (2 phrase), Err = {:.3f}, Time = {:.2f}'.format(np.sum(np.square(targets-results)),time_training))
    plt.show()
    return results
if __name__=='__main__':

    out_dim = 2
    num_ray = 10
    max_iters = 50
    step_size = 0.01
    criterion = "quadratic convex" # cosin complex, weighted distance, utility, linear, quadratic convex
    type = "2 phrases" # "1 phrase"
    threshold = 0.1
    alpha = 0.5
    hidden_dim = 200
    lr = 1e-3
    wd = 0.
    epochs = 2000
    alpha_r = 0.3
    n_tasks = 2

    sol, time_training = train(device = device, hidden_dim = hidden_dim,
    lr = lr, wd = wd, epochs = epochs, alpha = alpha, alpha_r = alpha_r, outdim = out_dim,
    step_size = step_size, max_iters = max_iters, criterion = criterion,
    type = type, threshold = threshold, n_tasks = n_tasks)
    x = []
    y = []
    for s in sol:
        x.append(f_1(torch.Tensor([s])).item())
        y.append(f_2(torch.Tensor([s])).item())
    plt.scatter(pf[:,0],pf[:,1],s=10,c='gray')
    plt.scatter(x[0],y[0], c = 'y', s = 80,label="Initial Point")
    plt.scatter(x[1:],y[1:], c = 'red', s = 30)
    plt.title('Pareto front TC02 (2 phrase)')
    predict(time_training,criterion)

    # sol_total,sol_total1 = train1(device = get_device(no_cuda=args.no_cuda, gpus=args.gpus), hidden_dim=200,
    #               lr=1e-4, wd=0., epochs=500, alpha=0.3,outdim=outdim,num_ray = num_ray)
    # # # plt.figure(1)
    # # # init_x,init_y = [],[]
    # # # for i in range(num_ray):
    # # #     sols = np.array(sol_total1[i])
    # # #     #print(sols)
    # # #     x = []
    # # #     y = []
    # # #     for s in sols:
    # # #         x.append(toy_loss_1(torch.Tensor([s])).item())
    # # #         y.append(toy_loss_2(torch.Tensor([s])).item())
    # # #     #colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, 1000))
    # # #     plt.plot(pf[:,0],pf[:,1],c='gray')
    # # #     plt.scatter(x[0],y[0], c = 'y', s = 80,label="Initial Point")
    # # #     #plt.scatter(x[1:-1],y[1:-1], c= 'black', s = 30)
    # # #     plt.plot(x[0:],y[0:])
    # # #     plt.scatter(x[-1],y[-1], c = 'red', s = 50)
    # # #     #plt.scatter(x[1:],y[1:],c='r',s=10)
    # # #     #plt.plot(x[0:],y[0:])
    # # #     init_x.append(x[0])
    # # #     init_y.append(y[0])
    # # # plt.savefig("trajectory1.png")
    # # # plt.figure(2)
    # for i in range(num_ray):
    #     sols = np.array(sol_total[i])
    #     #print(sols)
    #     x = []
    #     y = []
    #     # x.append(init_x[i])
    #     # y.append(init_y[i])
    #     for s in sols:
    #         x.append(toy_loss_1(torch.Tensor([s])).item())
    #         y.append(toy_loss_2(torch.Tensor([s])).item())
    #     #colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, 1000))
    #     plt.plot(pf[:,0],pf[:,1],c='gray')
    #     #plt.scatter(pf[:,0],pf[:,1],s=10,c='gray',label='Pareto Front',zorder=0)
    #     plt.scatter(x[0],y[0], c = 'y', s = 60,label="Initial Point 1")
    #     #plt.plot(x[0:2],y[0:2],'--')
    #     #plt.scatter(x[1],y[1], c = 'blue', s = 60,label="Initial Point 2")
    #     #plt.scatter(x[1:-1],y[1:-1], c= 'black', s = 30)
    #     plt.plot(x[1:],y[1:])
    #     plt.scatter(x[-1],y[-1], c = 'red', s = 30)
    #     #plt.scatter(x[1:],y[1:],c='r',s=10)
    #     #plt.plot(x[0:],y[0:])
    # # plt.savefig("trajectory.png")
    # plt.plot(pf[:,0],pf[:,1],c='gray',label='Pareto Front',zorder=0)
    #plt.show()
