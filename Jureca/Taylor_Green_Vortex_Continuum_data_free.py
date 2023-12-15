import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from torch import nn
import argparse
import sys, os, time, random, shutil
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader

def pars_ini():
    global args
    #
    parser = argparse.ArgumentParser(description='PyTorch actuated')

    #IO
    parser.add_argument('--data-dir', default='./', help='location of the training dataset')
    parser.add_argument('--restart-int', type=int, default=10, help='restart interval per epoch (default: 10)')
    parser.add_argument('--test_ID', type=str, default='0', help='Run iteration based on training data')

    #model
    parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=72, help='number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--wdecay', type=float, default=0.003, help='weight decay in ADAM (default: 0.003)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma in schedular (default: 0.95)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')
    parser.add_argument('--train_percent', type=int, default=0, help='percent data from domain used for training with ground truth')
    parser.add_argument('--model_type', type=str, default='PINN', help='define model type')
    
    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=200,
                        help='log interval per training')

    # parallel parsers
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--nworker', type=int, default=0,
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    args = parser.parse_args()

class Taylor_green_vortex_PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        self.layers = layers

        # layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self,X):
        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X)


        #x = scaling(X)
        x = X
        # convert to float
        a = x.float()

        '''     
            Alternatively:

            a = self.activation(self.fc1(a))
            a = self.activation(self.fc2(a))
            a = self.activation(self.fc3(a))
            a = self.fc4(a)

            '''
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)

            a = self.activation(z)

        a = self.linears[-2](a)        

        a = self.activation(a)

        a = self.linears[-1](a)

        #print("\tIn Model: input size", X.size(), "output size", a.size())

        return a

def scaling(X):

    '''mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
    # preprocessing input
    x = (X - mean) / (std)  # feature scaling'''
    x = X[:,0:2]
    t = X[:,2]
    t = torch.reshape(t, (t.shape[0],1))
    t = t/100 
    min_x, max_x = torch.min(x), torch.max(x)
    # preprocessing input
    x = -1 + 2*((x - min_x) / (max_x - min_x))
    
    X1 = torch.hstack([x,t])


    return X1

def h5_loader_initial(path):
    h5 = h5py.File(path, 'r')

    try:
        domain = h5.get('domain')
        left = h5.get('left')
        right = h5.get('right')
        top = h5.get('top')
        bottom = h5.get('bottom')
        full = h5.get('full')

        train_domain = np.array(domain.get('data1'))
        test_domain = np.array(domain.get('data2'))

        train_left = np.array(left.get('data1'))
        test_left = np.array(left.get('data2'))

        train_right = np.array(right.get('data1'))
        test_right = np.array(right.get('data2'))

        train_top = np.array(top.get('data1'))
        test_top = np.array(top.get('data2'))

        train_bottom = np.array(bottom.get('data1'))
        test_bottom = np.array(bottom.get('data2'))

        X_in = np.array(full.get('data1'))
        V_p_in = np.array(full.get('data2'))
        p_max = np.array(full.get('data3'))
        p_min = np.array(full.get('data4'))
        X_val = np.array(full.get('data5'))
        X_data = np.array(full.get('data6'))
        #print(X_domain.shape)
        #print(train_left[:,0:3].shape)
        #print(train_top[:,0:3].shape)
        #print(train_right[:,0:3].shape)
        #print(train_bottom[:,0:3].shape)
        p_max = torch.from_numpy(p_max).float()
        p_min = torch.from_numpy(p_min).float()

        '''print(X_train_domain.shape)
        print(X_train_left.shape)
        print(X_train_right.shape)
        print(X_train_top.shape)
        print(X_train_bottom.shape)
        print(V_p_train_domain.shape)
        print(V_p_train_left.shape)
        print(V_p_train_right.shape)
        print(V_p_train_top.shape)
        print(V_p_train_bottom.shape)'''

        len_data = len(np.vstack([train_left, train_right, train_top, train_bottom]))
        data_train = np.vstack([train_domain, train_left, train_right, train_top, train_bottom])
        idx = np.random.choice(data_train.shape[0], len_data, replace=False)

        train_data = data_train[idx,:]

        #train_physical = np.vstack([X_data, train_left[:,0:3], train_right[:,0:3], train_top[:,0:3], train_bottom[:,0:3]])
        test_data = np.vstack([test_domain, test_left, test_right, test_top, test_bottom])

        #print('train', train_physical.shape)
        #print('test', test_data.shape)
        '''print('########################################')
        for i in range(len(train_data)):
            print(train_data[i])
            print("\n")
        print('#######################################')
        '''
        #plt.tricontourf(train_data[:,0], train_data[:,1], train_data[:,4], levels=7)
        #plt.show()
    except Exception as e:
        print(e)

    return train_data,X_data,X_val, test_data, X_in, V_p_in, p_max, p_min


def h5_loader(path):
    h5 = h5py.File(path, 'r')

    try:
        domain = h5.get('domain')
        left = h5.get('left')
        right = h5.get('right')
        top = h5.get('top')
        bottom = h5.get('bottom')
        full = h5.get('full')

        train_domain = np.array(domain.get('data1'))
        test_domain = np.array(domain.get('data2'))

        train_left = np.array(left.get('data1'))
        test_left = np.array(left.get('data2'))

        train_right = np.array(right.get('data1'))
        test_right = np.array(right.get('data2'))

        train_top = np.array(top.get('data1'))
        test_top = np.array(top.get('data2'))

        train_bottom = np.array(bottom.get('data1'))
        test_bottom = np.array(bottom.get('data2'))

        X_in = np.array(full.get('data1'))
        V_p_in = np.array(full.get('data2'))
        p_max = np.array(full.get('data3'))
        p_min = np.array(full.get('data4'))
        X_val = np.array(full.get('data5'))
        X_data = np.array(full.get('data6'))
        #print(X_domain.shape)
        #print(train_left[:,0:3].shape)
        #print(train_top[:,0:3].shape)
        #print(train_right[:,0:3].shape)
        #print(train_bottom[:,0:3].shape)
        p_max = torch.from_numpy(p_max).float()
        p_min = torch.from_numpy(p_min).float()

        '''print(X_train_domain.shape)
        print(X_train_left.shape)
        print(X_train_right.shape)
        print(X_train_top.shape)
        print(X_train_bottom.shape)
        print(V_p_train_domain.shape)
        print(V_p_train_left.shape)
        print(V_p_train_right.shape)
        print(V_p_train_top.shape)
        print(V_p_train_bottom.shape)'''

        train_data = np.vstack([train_left, train_right, train_top, train_bottom])
        #train_physical = np.vstack([X_data, train_left[:,0:3], train_right[:,0:3], train_top[:,0:3], train_bottom[:,0:3]])
        test_data = np.vstack([test_domain, test_left, test_right, test_top, test_bottom])

        #print('train', train_physical.shape)
        #print('test', test_data.shape)
        '''print('########################################')
        for i in range(len(train_data)):
            print(train_data[i])
            print("\n")
        print('#######################################')
        '''
        #plt.tricontourf(train_data[:,0], train_data[:,1], train_data[:,4], levels=7)
        #plt.show()
    except Exception as e:
        print(e)

    return train_data,X_data,X_val, test_data, X_in, V_p_in, p_max, p_min

def result_loader(path):

    data = data9.get('result')
    x_val = np.array(data.get('x_val'))
    pred_val = np.array(data.get('pred_val'))

    data_val = np.hstack([x_val, pred_val])
    return data_val

def denormalize(p_norm, p_max, p_min):
    p = (((p_norm + 1) * (p_max - p_min)) / 2) + p_min
    return p


class GenerateDataset(Dataset):

    def __init__(self, list_id, path, trained_step,per):
        self.train_data,_,_ , _, _,_ ,_= h5_loader(path)
        
        for t in trained_step:
            path1 = './data/data_Taylor_Green_Vortex_'+str(per)+'_0.h5'
            if t==0:
                self.train_data_i0,_,_,_,_,_,_,_ = h5_loader(path1)
            else:
                path2 = './result/result_Taylor_Green_Vortex_0'+str(t)+'_20000.h5' 
                #print('else')
                self.train_data_i= result_loader(path2)
                self.train_data_i0 = np.hstack([self.train_data_i0,self.train_data_i])
        #print('train', self.train_data_i0.shape)         
        self.len = len(self.train_data)

    def __getitem__(self,index):
        data0 = self.train_data[index]
        datai = self.train_data_i0[index]
        data = torch.from_numpy(data0).float()
        data1 = torch.from_numpy(datai).float()
        return data, data1

    def __len__(self):
        return self.len   

class InitialDataset(Dataset):
    def __init__(self, list_id, path, per):
        self.train_data,_,_,_,_,_,_ ,_= h5_loader(path)
        self.len = len(self.train_data)

        path1 = './data/data_Taylor_Green_Vortex_'+str(per)+'_0.h5'
        self.train_data0,_,_ ,_, _, _,_ ,_= h5_loader_initial(path1)

    def __getitem__(self,index):
        data0 = self.train_data[index]
        datai = self.train_data0[index]
        data = torch.from_numpy(data0).float()
        data1 = torch.from_numpy(datai).float()
        return data, data1

    def __len__(self):
        return self.len

class PhysicalDataset(Dataset):
    def __init__(self, list_id, path):
        _,self.train_physical,_,_,_,_,_,_ = h5_loader(path)
        self.len = len(self.train_physical)

    def __getitem__(self, index):
        data0 = self.train_physical[index]
        data = torch.from_numpy(data0).float()
        return data
    
    def __len__(self):
        return self.len

class TestDataset(Dataset):

    def __init__(self, list_id, path, trained_step,per):
        _,_, self.test_data, _, _,_, _ ,_= h5_loader(path)
        for t in trained_step:
            path1 = './data/data_Taylor_Green_Vortex_'+str(per)+'_'+str(t)+'.h5'
            if t==0:
                _,_,_, self.test_data_i0,_,_,_,_ = h5_loader(path1)
            else:
                _,_,_, self.test_data_i,_,_,_,_ = h5_loader(path1)
                self.test_data_i0 = np.hstack([self.test_data_i0,self.test_data_i])
        #print('test', self.test_data_i0.shape)
        self.len = len(self.test_data)

    def __getitem__(self, index):
        data0 = self.test_data[index]
        datai = self.test_data_i0[index]
        data = torch.from_numpy(data0).float()
        data1 = torch.from_numpy(datai).float()
        return data, data1
    
    def __len__(self):
        return self.len

def train(model, device , train_loader,physical_loader, optimizer, epoch,grank, gwsize, rho, mu, p_max, p_min, trained_step):
    model.train()
    t_list = []
    loss_acc = 0
    if grank==0:
        print("\n")
    count = 0
    phy_data = iter(physical_loader)
    for batch_idx, (data, data1) in enumerate(train_loader):
        t = time.perf_counter()
        #if count % 1000 == 0 and grank==0:
           #print('Batch: ', count)
        optimizer.zero_grad()
        # predictions = distrib_model(inputs)
        #with torch.cuda.amp.autocast():
        physical_data = next(phy_data)
        loss = total_loss(model, data,physical_data,data1, device, rho, mu, p_max, p_min, grank, trained_step)
        
        loss.backward()
        
        optimizer.step()

        if batch_idx % args.log_int == 0 and grank==0:
            print('Batch: ', count)
            print(f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)/gwsize}'
                  f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\tLoss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
        loss_acc += loss.item()

        count += 1
    if grank==0:
        print('TIMER: train time', sum(t_list) / len(t_list), 's')

    return loss_acc / len(train_loader.dataset)


def test(model, device, test_loader, grank, gwsize, rho, mu, trained_step):
    loss_function = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    test_err_i = 0.0
    rel_err = 0.0
    #test_loss_acc = []
    with torch.no_grad():
        
        for data, data1 in test_loader:
            # print(data)
            initial_err = torch.zeros((1)).to(device) 
            for i in range(0, len(trained_step)):
                k = 6*i
                inputs_i = data1[:,k:k+3].to(device)
                exact_i = data1[:,k+3:k+6].to(device)
                outputs_i = model(inputs_i)
                initial_err += torch.linalg.norm((exact_i - outputs_i[:,0:3]), 2)/torch.linalg.norm(exact_i,2)
            
            inputs = data[:,0:3].to(device)
            outputs = model(inputs)
            exact = data[:,3:6].to(device)
            loss = loss_function(outputs[:,0:3], exact)
            error = torch.linalg.norm((exact - outputs[:,0:3]), 2) / torch.linalg.norm(exact,2)
            
            test_loss += loss.item() 
            rel_err += error.item()
            test_err_i += initial_err.item()

    return test_loss/(len(test_loader.dataset)), rel_err/(len(test_loader.dataset)), test_err_i/(len(test_loader.dataset))

def denormalize_full(V_p_norm, p_min, p_max):
    u_norm = V_p_norm[:,0]
    v_norm = V_p_norm[:,1]
    p_norm = V_p_norm[:,2]

    #u = ((u_norm + 1) * (u_max - u_min) / 2) + u_min
    #v = ((v_norm + 1) * (v_max - v_min) / 2) + v_min
    p = ((p_norm + 1) * (p_max - p_min) / 2) + p_min

    return u_norm, v_norm, p_norm

# save state of the training
def save_state(i,epoch,distrib_model,loss_acc,optimizer,res_name, grank, gwsize, is_best, loss_acc_list, rel_error_list, initial_err_list, trained_step):#,grank,gwsize,is_best):
    rt = time.time()
    # find if is_best happened in any worker
    is_best_m = par_allgather_obj(is_best,gwsize)

    if any(is_best_m):
        # find which rank is_best happened - select first rank if multiple
        is_best_rank = np.where(np.array(is_best_m)==True)[0][0]

        # collect state
        state = {'epoch': epoch + 1,
                 'start': i,
                'state_dict': distrib_model.state_dict(),
                'best_acc': loss_acc,
                'optimizer' : optimizer.state_dict(),
                'loss_acc_list': loss_acc_list,
                'rel_error_list': rel_error_list,
                'initial_err_list': initial_err_list,
                'trained_step': trained_step}

        # write on worker with is_best
        if grank == is_best_rank:
            torch.save(state,'./'+res_name)
            print(f'DEBUG: state is saved on epoch:{epoch} in {time.time()-rt} s')

# deterministic dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# PARALLEL HELPERS
# sum of field over GPGPUs
def par_sum(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    return res

# mean of field over GPGPUs
def par_mean(field,gwsize):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    res/=gwsize
    return res

# max(field) over GPGPUs
def par_max(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.MAX,group=None,async_op=True).wait()
    return res

# min(field) over GPGPUs
def par_min(field):
    res = torch.tensor(field).float()
    res = res.cuda() if args.cuda else res.cpu()
    dist.all_reduce(res,op=dist.ReduceOp.MIN,group=None,async_op=True).wait()
    return res

# reduce field to destination with an operation
def par_reduce(field,dest,oper):
    '''
    dest=0 will send the result to GPU on rank 0 (any rank is possible)
    op=oper has to be in form "dist.ReduceOp.<oper>", where <oper> is
      SUM
      PRODUCT
      MIN
      MAX
      BAND
      BOR
      BXOR
    '''
    res = torch.Tensor([field])
    res = res.cuda() if args.cuda else res.cpu()
    dist.reduce(res,dst=dest,op=oper,group=None,async_op=False)
    return res

# gathers tensors from the whole group in a list (to all workers)
def par_allgather(field,gwsize):
    if args.cuda:
        sen = torch.Tensor([field]).cuda()
        res = [torch.Tensor([field]).cuda() for i in range(gwsize)]
    else:
        sen = torch.Tensor([field])
        res = [torch.Tensor([field]) for i in range(gwsize)]
    dist.all_gather(res,sen,group=None)
    return res

# gathers any object from the whole group in a list (to all workers)
def par_allgather_obj(obj,gwsize):
    res = [None]*gwsize
    dist.all_gather_object(res,obj,group=None)
    return res

def total_loss(model, data,physical_data, data1, device, rho, mu, p_min, p_max, grank, trained_step):

    loss_function = nn.MSELoss()
    #print(data)
    loss_initial = torch.zeros((1)).to(device)
    for i in range(0, len(trained_step)):
        k = 6*i
        inputs_i = data1[:,k:k+3].to(device)
        exact_i = data1[:,k+3:k+6].to(device)
        output_i = model(inputs_i)
        loss_initial += loss_function(output_i[:,0:3], exact_i)

    inputs = data[:,0:3].to(device)
    phy_input = physical_data.to(device)
    if grank==0:
        print("phy_input", phy_input)
        print("inputs", inputs)
    #flag = data[:6]
    #print('input', inputs)
    
    exact = data[:,3:6].to(device)

    #print('exact', exact)
    g1 = inputs.clone()
    g1.requires_grad = True
    
    output_data = model(g1)


    g = phy_input.clone()
    g.requires_grad = True
   
    #global output
    output = model(g)
    #output_i = model(inputs_i)
         
    u = output[:,0]
    v = output[:,1]
    p = output[:,2]
    s11 = output[:,3]
    s22 = output[:,4]
    s12 = output[:,5]

    '''
    x = inputs[:,0]
    y = inputs[:,1]
    t = inputs[:,2]

    x = (torch.reshape(x, (x.shape[0],1)),)
    y = (torch.reshape(y, (y.shape[0],1)),)
    t = (torch.reshape(t, (t.shape[0],1)),)

    '''
    
    u = torch.reshape(u, (u.shape[0],1))
    v = torch.reshape(v, (v.shape[0],1))
    p = torch.reshape(p, (p.shape[0],1))

    #p = denormalize(p_norm, p_max, p_min)
    s11 = torch.reshape(s11, (s11.shape[0],1))
    s22 =  torch.reshape(s22, (s22.shape[0],1))
    s12 = torch.reshape(s12, (s12.shape[0],1))

    u_x_y_t = torch.autograd.grad(u,g,torch.ones([phy_input.shape[0], 1]).to(device), create_graph=True)[0]
    v_x_y_t = torch.autograd.grad(v,g,torch.ones([phy_input.shape[0], 1]).to(device), create_graph=True)[0]
    s_11_x_y_t = torch.autograd.grad(s11,g,torch.ones([phy_input.shape[0], 1]).to(device), create_graph=True)[0]
    s_12_x_y_t = torch.autograd.grad(s12,g,torch.ones([phy_input.shape[0], 1]).to(device), create_graph=True)[0]
    s_22_x_y_t = torch.autograd.grad(s22,g,torch.ones([phy_input.shape[0], 1]).to(device), create_graph=True)[0]

    #if grank == 0:
        #print('uxyt', u_x_y_t.shape)
    
    ux = u_x_y_t[:,0]
    uy = u_x_y_t[:,1]
    ut = u_x_y_t[:,2]

    vx = v_x_y_t[:,0]
    vy = v_x_y_t[:,1]
    vt = v_x_y_t[:,2]

    s11_x = s_11_x_y_t[:,0]
    s22_y = s_22_x_y_t[:,1]
    s12_x = s_12_x_y_t[:,0]
    s12_y = s_12_x_y_t[:,1]

    '''
    ux = torch.gradient(u, spacing = 0.1, edge_order=2)[0]
    uy = torch.gradient(u, spacing = 0.1, edge_order=2)[0]
    ut = torch.gradient(u, spacing = 1, edge_order=2)[0]
    
    vx = torch.gradient(v, spacing = 0.1, edge_order=2)[0]
    vy = torch.gradient(v, spacing = 0.1, edge_order=2)[0]
    vt = torch.gradient(v, spacing = 1, edge_order=2)[0]

    s11_x = torch.gradient(s11, spacing = 0.1, edge_order=2)[0]
    s22_y = torch.gradient(s22, spacing = 0.1, edge_order=2)[0]

    s12_x = torch.gradient(s12, spacing = 0.1, edge_order=2)[0]
    s12_y = torch.gradient(s12, spacing = 0.1, edge_order=2)[0]  
    '''

    ut =  torch.reshape(ut, (ut.shape[0],1))
    vt =  torch.reshape(vt, (vt.shape[0],1))
    ux = torch.reshape(ux, (ux.shape[0],1))
    uy = torch.reshape(uy, (uy.shape[0],1))
    vx = torch.reshape(vx, (vx.shape[0],1))
    vy = torch.reshape(vy, (vy.shape[0],1))

    s11_x = torch.reshape(s11_x, (s11_x.shape[0],1))
    s22_y =  torch.reshape(s22_y, (s22_y.shape[0],1))
    s12_x = torch.reshape(s12_x, (s12_x.shape[0],1))
    s12_y = torch.reshape(s12_y, (s12_y.shape[0],1))

    continuity = ux + vy
    #ns1 = ut + torch.mul(u,ux) + torch.mul(v,uy) + (1/rho)*px*(p_max - p_min)/2 - nu*(u_xx + u_yy)
    #ns2 = vt + torch.mul(u,vx) + torch.mul(v,vy) + (1/rho)*py*(p_max - p_min)/2 - nu*(v_xx + v_yy)
    
    fu = rho*ut + rho*(torch.mul(u,ux) + torch.mul(v,uy)) - s11_x - s12_y
    fv = rho*vt + rho*(torch.mul(u,vx) + torch.mul(v,vy)) - s12_x - s22_y
        
    f_s11 = - p + 2*mu*ux - s11
    f_s22 = - p + 2*mu*vy - s22
    f_s12 = mu*(uy+vx) - s12
    
    '''if grank==0:
        print('first', continuity.shape)
        print('second',fu.shape)
        print('second',fv.shape)
        print('second',f_s11.shape)
        print('second',f_s22.shape)
        print('second',f_s12.shape)
    '''

    target1 = torch.zeros_like(continuity, device=device)
    target2 = torch.zeros_like(fu, device=device)
    target3 = torch.zeros_like(fv, device=device)
    target4 =  torch.zeros_like(f_s11, device=device)
    target5 =  torch.zeros_like(f_s22, device=device)
    target6 =  torch.zeros_like(f_s12, device=device)

    loss_continuity = loss_function(continuity, target1)
    loss_fu = loss_function(fu, target2)
    loss_fv = loss_function(fv, target3)
    loss_fs11 = loss_function(f_s11, target4)
    loss_fs22 = loss_function(f_s22, target5)
    loss_fs12 = loss_function(f_s12, target6)

    '''loss_variable = torch.zeros((1,), requires_grad=True, device=device)
    for i in range(0,data.shape[0]):
        if data[i,6] == 0:
            loss_variable = loss_variable.clone().to(device)
            loss_variable += loss_function(output[i,0:3], exact[i])
        
    if grank==0:
        print("loss_var",loss_variable)
    '''
    loss_variable = loss_function(output_data[:,0:3], exact)
    #print("\tloss_continuity :", loss_continuity, "loss_momentum1 :", loss_ns1, "loss_momentum2 :", loss_ns2, "loss_variable: ", loss_variable)

    loss =  loss_continuity + loss_fu + loss_fv + loss_fs11 + loss_fs22 + loss_fs12 + loss_initial + loss_variable

    #loss = loss_initial

    return loss

def main():

    # get parse arguments
    pars_ini()

    # check for CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # get directory
    program_dir = os.getcwd()

    torch.backends.cudnn.benchmark = True

    # start time
    st = time.time()

    # initialize distributed backend
    dist.init_process_group(backend=args.backend)

    # deterministic testrun
    if args.testrun:
        torch.manual_seed(args.seed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    #  get job rank
    lwsize = torch.cuda.device_count() if args.cuda else 0 # local world size - per node
    gwsize = dist.get_world_size() # global world size - per run
    grank = dist.get_rank() # global rank - assign per run
    lrank = dist.get_rank()%lwsize # local rank - assign per node

    if grank == 0:
        print('TIMER: initialise:', time.time() - st, 's')
        #print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:', sys.version, '\n')

        print('DEBUG: IO parsers:')
        print('DEBUG: args.data_dir:', args.data_dir)
        print('DEBUG: args.restart_int:', args.restart_int, '\n')

        print('DEBUG: model parsers:')
        print('DEBUG: args.batch_size:', args.batch_size)
        print('DEBUG: args.epochs:', args.epochs)
        print('DEBUG: args.lr:', args.lr)
        print('DEBUG: args.wdecay:', args.wdecay)
        print('DEBUG: args.gamma:', args.gamma)
        print('DEBUG: args.shuff:', args.shuff, '\n')

        print('DEBUG: debug parsers:')
        print('DEBUG: args.testrun:', args.testrun)
        print('DEBUG: args.nseed:', args.nseed)
        print('DEBUG: args.log_int:', args.log_int, '\n')

        print('DEBUG: parallel parsers:')
        print('DEBUG: args.backend:', args.backend)
        print('DEBUG: args.nworker:', args.nworker)
        print('DEBUG: args.prefetch:', args.prefetch)
        print('DEBUG: args.cuda:', args.cuda)
        print('DEBUG: args.benchrun:', args.benchrun, '\n')
        print('DEBUG: args.train_percent:', args.train_percent, '\n')
        print('DEBUG: args.test_ID:', args.test_ID, '\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        # deterministic testrun
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)


    rho = 1.0
    mu = 0.01

    per = args.train_percent

    if grank == 0:
        print('Percentage training data: ',per)
        print(f'TIMER: read data: {time.time() - st} s\n')

    # create model

    layers = np.array([3, 300, 300, 300, 300, 300, 6])
    model = Taylor_green_vortex_PINN(layers).to(device)

    print(device)
    # distribute model too workers


    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model, \
                                                                  device_ids=[device], output_device=device,
                                                                  find_unused_parameters=False)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

        # optimizer
    optimizer = torch.optim.SGD(distrib_model.parameters(), lr=args.lr)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scheduler = scheduler_cosine

    t = np.arange(0,30,5).tolist()
    t.append(30)
    #print(t)

    #ti = args.test_ID

    start = 0
    start_epoch=1
    best_acc = np.Inf
    loss_acc_list = []
    rel_error_list = []
    initial_err_list = []
    trained_step = []
    # resume state
    res_name = 'checkpoint_red0a.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(program_dir + '/' + res_name, map_location=loc)
            start = checkpoint['start']
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            loss_acc_list = checkpoint['loss_acc_list']
            rel_error_list = checkpoint['rel_error_list']
            initial_err_list = checkpoint['initial_err_list']
            distrib_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            trained_step = checkpoint['trained_step']
            if grank == 0:
                print(f'WARNING: restarting from time={t[start+1]} and {start_epoch} epoch')
        except Exception as e:
            if grank == 0:
                print(e)
                print(f'WARNING: restart file cannot be loaded, restarting!')


    for i in range(start,len(t)-1):

        if start_epoch == args.epochs +1:
            start_epoch = 1
            continue

        elif start_epoch > args.epochs +1:
            if grank == 0:
                print(f'WARNING: given epochs are less than the one in the restart file!\n'
                      f'WARNING: SYS.EXIT is issued')
            dist.destroy_process_group()
            sys.exit()
        if t[i] not in trained_step:
            trained_step.append(t[i])

        if grank==0:
            print('Starting training for t=',t[i+1])

        path = './data/data_Taylor_Green_Vortex_'+str(per)+'_'+str(t[i+1])+'.h5'

        train_data,physical_data, val_data, test_data, X_in, V_p_star, p_max, p_min = h5_loader(path)

        X_in = torch.from_numpy(X_in).float().to(device)

        train_len = len(train_data)
        physical_len = len(physical_data)
        test_len = len(test_data)

        batches = int(train_len/args.batch_size)
        batch_size_physical = int(physical_len/batches)
        if grank==0:
            print('Trained steps: ', trained_step)
            print('No. of batches: ',batches)
            print('Physical data batch size', batch_size_physical)

        # restricts data loading to a subset of the dataset exclusive to the current process
        args.shuff = args.shuff and not args.testrun
        
        if t[i]==0:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset = InitialDataset([x for x in range(train_len)], path, per),
                                num_replicas=gwsize, rank=grank, shuffle=True)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset = GenerateDataset([x for x in range(train_len)], path,trained_step,per),
                                num_replicas=gwsize, rank=grank, shuffle=True)

        physical_sampler = torch.utils.data.distributed.DistributedSampler(dataset = PhysicalDataset([x for x in range(physical_len)], path),
                        num_replicas=gwsize, rank=grank, shuffle=True)
                
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset = TestDataset([x for x in range(test_len)], path, trained_step, per),
                                num_replicas=gwsize, rank=grank, shuffle=True)

        # distribute dataset to workers
        # persistent workers
        pers_w = True if args.nworker>1 else False
        
        # deterministic testrun - the same dataset each run
        kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}
        
        if t[i]==0:
            train_loader = torch.utils.data.DataLoader(dataset = InitialDataset([x for x in range(physical_len)], path,per),
                                                   batch_size=args.batch_size,
                                                   sampler = train_sampler,
                                                   num_workers=args.nworker, pin_memory=False,
                                                   persistent_workers=pers_w, drop_last=True,
                                                   prefetch_factor=args.prefetch, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(dataset = GenerateDataset([x for x in range(train_len)], path, trained_step,per), 
                                                   batch_size=args.batch_size,
                                                   sampler = train_sampler,
                                                   num_workers=args.nworker, pin_memory=False,
                                                   persistent_workers=pers_w, drop_last=True,
                                                   prefetch_factor=args.prefetch, **kwargs)
        
        physical_loader = torch.utils.data.DataLoader(dataset = PhysicalDataset([x for x in range(physical_len)], path), 
                                           batch_size=batch_size_physical,
                                           sampler = physical_sampler,
                                           num_workers=args.nworker, pin_memory=False,
                                           persistent_workers=pers_w, drop_last=True,
                                           prefetch_factor=args.prefetch, **kwargs)

        test_loader = torch.utils.data.DataLoader(dataset = TestDataset([x for x in range(test_len)], path, trained_step,per), batch_size=2,
                                          sampler=test_sampler, num_workers=args.nworker, pin_memory=False,
                                          persistent_workers=pers_w, drop_last=True,
                                          prefetch_factor=args.prefetch, **kwargs)


            

        # start training/testing loop
        if grank==0:
            print('TIMER broadcast:', time.time()-st, 's')
            print(f'\nDEBUG: start training')
            print(f'------------------------------------------')

        et = time.time()

        test_acc_list = []
        lr_list = []
        for epoch in range(start_epoch, args.epochs + 1):

            lt = time.time()

            #training
            if args.benchrun and epoch==args.epochs:
                with torch.autograd.profiler.profile(use_cuda=args.cuda, profile_memory=True) as prof:
                    loss_acc = train(distrib_model, device, train_loader,physical_loader, optimizer,epoch, grank, gwsize, rho, mu, p_max,p_min, trained_step)
            else:
                loss_acc = train(distrib_model, device, train_loader,physical_loader, optimizer, epoch, grank, gwsize, rho, mu, p_max, p_min, trained_step)

            loss_acc_list.append(loss_acc)

            # testing
            acc_test, rel_err, error_initial = test(distrib_model, device, test_loader, grank, gwsize, rho, mu, trained_step)

            # lr Scheduler
            #scheduler.step()

            latest_lr = scheduler.get_last_lr()

            if grank == 0 and lrank == 0:
                print('Time: ',t[i+1])
                print('Epoch finished')
                print('Epoch: {:03d}, Loss: {:.5f}, Test MSE: {:.5f}, Test Error: {:.5f}, Initial_Error: {:.5f}'.
                    format(epoch, loss_acc, acc_test, rel_err, error_initial))

            lr_list.append(latest_lr)

            test_acc_list.append(acc_test)
            rel_error_list.append(rel_err)
            initial_err_list.append(error_initial)

            if epoch == start_epoch:
                first_ep_t = time.time() - lt

            # final epoch
            if epoch + 1 == args.epochs:
                train_loader.last_epoch = True
                test_loader.last_epoch = True

            if grank==0:
                print(f'\n--------------------------------------------------------')
                print(f'TIMER: epoch time:', time.time()-lt, 's')
                print(f'DEBUG: accuracy:', acc_test, '%')
                if args.benchrun and epoch==args.epoch:
                    print(f'\n----------------------------------------')
                    print(f'DEBUG: benchmark of last epoch:\n')
                    what1 = 'cuda' if args.cuda else 'cpu'
                    print(prof.key_averages().table(sort_by='self_'+str(what1)+'_time_total'))

            # if a better state is found
            is_best = loss_acc < best_acc
            if epoch % args.restart_int == 0 and not args.benchrun:
                if grank==0:
                    print('Saving: ', res_name)
                save_state(i,epoch, distrib_model, loss_acc, optimizer, res_name, grank, gwsize, is_best, loss_acc_list, rel_error_list, initial_err_list, trained_step)
                #save_state(epoch, model, loss_acc, optimizer, res_name)
                best_acc = min(loss_acc, best_acc)
                g = X_in.clone()
                g_v = val_data.clone()
                pred_val = distrib_model(val_data)
                V_p_pred_norm = distrib_model(X_in)
                u_pred, v_pred, p_pred = denormalize_full(V_p_pred_norm[:,0:3], p_min, p_max)
                with torch.no_grad():
                    #V_p_star = V_p_star.cpu().detach().numpy()
                    g = g.cpu().detach().numpy()
                    g_v = g_v.cpu().detach().numpy()
                    pred_val = pred_val.cpu().detach().numpy()
                    #V_p_pred_norm = V_p_pred_norm.cpu().detach().numpy()
                    u_pred = u_pred.cpu().detach().numpy()
                    v_pred = v_pred.cpu().detach().numpy()
                    p_pred = p_pred.cpu().detach().numpy()
                    loss = np.asarray(loss_acc_list)
                    error = np.asarray(rel_error_list)
                    lr = np.asarray(lr_list)
                    initial_error = np.asarray(initial_err_list)
                    #result = [V_p_star,X_in,V_p_pred_norm, u_pred,v_pred, p_pred, loss_acc_list, rel_error_list, lr_list, initial_err_list]
                    if grank == 0:
                        h5 = h5py.File('./result/result_Taylor_Green_Vortex_0'+str(t[i+1])+'_'+str(epoch)+'.h5', 'w')
                        g1 = h5.create_group('result')
                        g1.create_dataset('star', data=V_p_star)
                        g1.create_dataset('X_in', data=g)
                        g1.create_dataset('u_pred', data=u_pred)
                        g1.create_dataset('v_pred', data=v_pred)
                        g1.create_dataset('p_pred', data=p_pred)
                        g1.create_dataset('loss', data=loss)
                        g1.create_dataset('error', data=error)
                        g1.create_dataset('lr', data=lr)
                        g1.create_dataset('initial', data=initial_error)
                        g1.create_dataset('x_val', data=g_v)
                        g1.create_dataset('pred_val', data=pred_val)
                        h5.close()
                '''if grank == 0:
                    print('Saving results at epoch: ',epoch)
                f = open('./result/result_Taylor_green_vortex_reduced'+str(t[i])+'_'+str(epoch)+'.pkl', 'wb')
                pickle.dump(result, f)
                f.close()'''

            if grank==0:
                print(epoch)

        start_epoch = 1

        #finalise training
        # save final state
        if not args.benchrun and epoch==args.epochs:
            save_state(i,epoch, distrib_model, loss_acc, optimizer, res_name, grank, gwsize, True, loss_acc_list, rel_error_list, initial_err_list, trained_step)
            #save_state(epoch, model, loss_acc, optimizer, res_name)
        dist.barrier()

        if grank==0:
            print('Sequence finsihed for t=',t[i+1])


    if grank == 0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: training results:\n')
        print(f'TIMER: first epoch time: {first_ep_t} s')
        print(f'TIMER: last epoch time: {time.time() - lt} s')
        print(f'TIMER: total epoch time: {time.time() - et} s')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time() - et} s')


    if grank==0:
        f_time = time.time() - st
        print(f'TIMER: final time: {f_time} s')

    # clean-up
    dist.destroy_process_group()

def test_model(t):
    
     # get parse arguments
    pars_ini()

    # check for CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # get directory
    program_dir = os.getcwd()

    torch.backends.cudnn.benchmark = True

    # start time
    st = time.time()

    # initialize distributed backend
    dist.init_process_group(backend=args.backend)

    # deterministic testrun
    if args.testrun:
        torch.manual_seed(args.seed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    #  get job rank
    lwsize = torch.cuda.device_count() if args.cuda else 0 # local world size - per node
    gwsize = dist.get_world_size() # global world size - per run
    grank = dist.get_rank() # global rank - assign per run
    lrank = dist.get_rank()%lwsize # local rank - assign per node

    if grank == 0:
        print('TIMER: initialise:', time.time() - st, 's')
        #print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:', sys.version, '\n')

        print('DEBUG: IO parsers:')
        print('DEBUG: args.data_dir:', args.data_dir)
        print('DEBUG: args.restart_int:', args.restart_int, '\n')

        print('DEBUG: model parsers:')
        print('DEBUG: args.batch_size:', args.batch_size)
        print('DEBUG: args.epochs:', args.epochs)
        print('DEBUG: args.lr:', args.lr)
        print('DEBUG: args.wdecay:', args.wdecay)
        print('DEBUG: args.gamma:', args.gamma)
        print('DEBUG: args.shuff:', args.shuff, '\n')

        print('DEBUG: debug parsers:')
        print('DEBUG: args.testrun:', args.testrun)
        print('DEBUG: args.nseed:', args.nseed)
        print('DEBUG: args.log_int:', args.log_int, '\n')

        print('DEBUG: parallel parsers:')
        print('DEBUG: args.backend:', args.backend)
        print('DEBUG: args.nworker:', args.nworker)
        print('DEBUG: args.prefetch:', args.prefetch)
        print('DEBUG: args.cuda:', args.cuda)
        print('DEBUG: args.benchrun:', args.benchrun, '\n')
    

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        # deterministic testrun
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

    lwsize = torch.cuda.device_count() if args.cuda else 0 # local world size - per node
    gwsize = dist.get_world_size() # global world size - per run
    grank = dist.get_rank() # global rank - assign per run
    lrank = dist.get_rank()%lwsize # local rank - assign per node

    layers = np.array([3, 300, 300, 300, 300, 300, 6])
    model = Taylor_green_vortex_PINN(layers).to(device)

    print(device)

    t_range = np.arange(0,30,5).tolist()
    t_range.append(30)
    # distribute model too workers
    

    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model, \
                                                                  device_ids=[device], output_device=device,
                                                                  find_unused_parameters=False)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    
    loss_acc_list = []
    rel_error_list = []
    initial_err_list = []
    lr_list = []
    start = 0
    start_epoch = 1
    best_acc = np.inf
    res_name = 'checkpoint_red0.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(program_dir + '/' + res_name, map_location=loc)
            start = checkpoint['start']
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            loss_acc_list = checkpoint['loss_acc_list']
            rel_error_list = checkpoint['rel_error_list']
            initial_err_list = checkpoint['initial_err_list']
            distrib_model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #trained_step = checkpoint['trained_step']
            if grank == 0:
                print(f'WARNING: restarting from time={t_range[start+1]} and {start_epoch} epoch')
        except Exception as e:
            if grank == 0:
                print(e)
                print(f'WARNING: restart file cannot be loaded, restarting!')
     
    path = './data/data_Taylor_Green_Vortex_reduced10'+'_'+str(t)+'.h5'

    train_data,physical_data, test_data, X_in, V_p_star, p_max, p_min = h5_loader(path)

    g = torch.from_numpy(X_in).float().to(device)

    V_p_pred_norm = distrib_model(g)
    u_pred, v_pred, p_pred = denormalize_full(V_p_pred_norm[:,0:3], p_min, p_max)

    u_pred = u_pred.cpu().detach().numpy()
    v_pred = v_pred.cpu().detach().numpy()
    p_pred = p_pred.cpu().detach().numpy()

    #print(u_pred)
    #print(v_pred)
    #print(p_pred)
    #print(X_in)
    if grank == 0:
    
        h5 = h5py.File('./result/test_Taylor_Green_Vortex_reduced0'+str(t)+'.h5', 'w')
        g1 = h5.create_group('result')
        g1.create_dataset('star', data=V_p_star)
        g1.create_dataset('X_in', data=X_in)
        g1.create_dataset('u_pred', data=u_pred)
        g1.create_dataset('v_pred', data=v_pred)
        g1.create_dataset('p_pred', data=p_pred)
        h5.close()
    
        print('Testing of model finished')                
    
    

if __name__ == "__main__":
    main()
    sys.exit()

#test_model(10)
    
#h5_loader()
