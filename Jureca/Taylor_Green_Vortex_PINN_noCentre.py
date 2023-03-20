#import matplotlib.pyplot as plt
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
from lion_pytorch import Lion

def pars_ini():
    global args

    parser = argparse.ArgumentParser(description='PyTorch actuated')

    #IO
    parser.add_argument('--data-dir', default='./', help='location of the training dataset')
    parser.add_argument('--restart-int', type=int, default=10, help='restart interval per epoch (default: 10)')

    #model
    parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=72, help='number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--wdecay', type=float, default=0.003, help='weight decay in ADAM (default: 0.003)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma in schedular (default: 0.95)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')

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
    parser.add_argument('--solution-int', type=int, default=50, help='saving prediction result after interval (default: 50)')

    args = parser.parse_args()

class Taylor_green_vortex_PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        # Activation
        self.activation = nn.Tanh()
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.activation3 = nn.SiLU()
        self.layers = layers

        # layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self,X):
        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X)


        x = scaling(X)
        # convert to float
        a = x.float()

        '''     
            Alternatively:

            a = self.activation(self.fc1(a))
            a = self.activation(self.fc2(a))
            a = self.activation(self.fc3(a))
            a = self.fc4(a)

            '''
        for i in range(len(self.layers) - 3):
            z = self.linears[i](a)

            a = self.activation3(z)

        a = self.linears[-2](a)        

        a = self.activation(a)

        a = self.linears[-1](a)

        #print("\tIn Model: input size", X.size(), "output size", a.size())

        return a

def scaling(X):

    mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
    # preprocessing input
    x = (X - mean) / (std)  # feature scaling

    return x

def h5_loader():
    h5 = h5py.File('./data/data_Taylor_Green_Vortex_reduced.h5', 'r')

    try:
        domain = h5.get('domain')
        left = h5.get('left')
        right = h5.get('right')
        top = h5.get('top')
        bottom = h5.get('bottom')
        initial = h5.get('initial')
        center = h5.get('center')
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

        train_initial = np.array(initial.get('data1'))
        test_initial = np.array(initial.get('data2'))

        center_data = np.array(center.get('data1'))

        X_in = np.array(full.get('data1'))
        V_p_star = np.array(full.get('data2'))

        #print(V_p_star)

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


        train_data = np.vstack([train_domain, train_left, train_right, train_top, train_bottom, train_initial])
        test_data = np.vstack([test_domain, test_left, test_right, test_top, test_bottom, test_initial])
        

        '''print('########################################')
        for i in range(len(train_data)):
            print(train_data[i])
            print("\n")

        print('#######################################')
        '''
    except Exception as e:
        print(e)

    return train_data, test_data, X_in, V_p_star

class GenerateDataset(Dataset):

    def __init__(self, list_id):
        self.train_data, _ , _, _ = h5_loader()
        self.len = len(self.train_data)

    def __getitem__(self,index):
        data0 = self.train_data[index]
        data = torch.from_numpy(data0).float()
        return data

    def __len__(self):
        return self.len    

class TestDataset(Dataset):

    def __init__(self, list_id):
        _, self.test_data, _, _ = h5_loader()
        self.len = len(self.test_data)

    def __getitem__(self, index):
        data0 = self.test_data[index]
        data = torch.from_numpy(data0).float()
        return data
    
    def __len__(self):
        return self.len

def train(model, device , train_loader, optimizer, epoch,grank, gwsize, rho, nu):
    model.train()
    t_list = []
    loss_acc = 0
    if grank==0:
        print("\n")
    count = 0
    for batch_idx, (data) in enumerate(train_loader):
        t = time.perf_counter()
        if count % 1000 == 0 and grank==0:
           print('Batch: ', count)
        optimizer.zero_grad()
        # predictions = distrib_model(inputs)
        #with torch.cuda.amp.autocast():
        loss = total_loss(model, data, device, rho, nu, epoch, batch_idx, grank)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_int == 0 and grank==0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)/gwsize}'
                  f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\tLoss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
        loss_acc += loss.item()

        count += 1
    if grank==0:
        print('TIMER: train time', sum(t_list) / len(t_list), 's')

    return loss_acc


def test(model, device, test_loader, grank, gwsize, rho, nu, epoch):
    model.eval()
    test_loss = 0.0

    test_loss_acc = []
    with torch.no_grad():
        for data in test_loader:
            # print(data)

            inputs = data[0]

            loss = total_loss(model, data, device, rho, nu, epoch, None, grank)

            test_loss += loss.item() / inputs.shape[0]

            # count+=1
            test_loss_acc.append(test_loss)

    return test_loss_acc

def denormalize(V_p_norm, u_min, u_max, v_min, v_max, p_min, p_max):
    u_norm = V_p_norm[:,0]
    v_norm = V_p_norm[:,1]
    p_norm = V_p_norm[:,2]

    u = ((u_norm + 1) * (u_max - u_min) / 2) + u_min
    v = ((v_norm + 1) * (v_max - v_min) / 2) + v_min
    p = ((p_norm + 1) * (p_max - p_min) / 2) + p_min

    return u, v, p

def denormalize01(V_p_norm, u_min, u_max, v_min, v_max, p_min, p_max):
    u_norm = V_p_norm[:,0]
    v_norm = V_p_norm[:,1]
    p_norm = V_p_norm[:,2]

    u = u_norm * (u_max - u_min) + u_min
    v = v_norm * (v_max - v_min) + v_min
    p = p_norm * (p_max - p_min) + p_min

    return u,v,p

# save state of the training
def save_state(epoch,distrib_model,loss_acc,optimizer,res_name, grank, gwsize, is_best):#,grank,gwsize,is_best):
    rt = time.time()
    # find if is_best happened in any worker
    is_best_m = par_allgather_obj(is_best,gwsize)

    if any(is_best_m):
        # find which rank is_best happened - select first rank if multiple
        is_best_rank = np.where(np.array(is_best_m)==True)[0][0]

        # collect state
        state = {'epoch': epoch + 1,
                'state_dict': distrib_model.state_dict(),
                'best_acc': loss_acc,
                'optimizer' : optimizer.state_dict()}

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

def prediction(x,y,t):
    g = torch.cat((x, y, t), dim=1)
    predictions = NNmodel(g)
    return predictions

def pred_hessian_u(x,y,t):
    g = torch.cat((x, y, t), dim=1)
    predictions = NNmodel(g)[:,0]
    return predictions.sum()

def pred_hessian_v(x,y,t):
    g = torch.cat((x, y, t), dim=1)
    predictions = NNmodel(g)[:,1]
    return predictions.sum()

def pred_hessian_p(x,y,t):
    g = torch.cat((x, y, t), dim=1)
    predictions = NNmodel(g)[:,2]
    return predictions.sum()

def total_loss(model, data, device, rho, nu, epoch, batch, grank):

    loss_function = nn.MSELoss()
    #print(data)
    inputs = data[:,0:3].to(device)

    #print('input', inputs)

    exact = data[:,3:6].to(device)

    #print('exact', exact)

    g = inputs.clone()
    g.requires_grad = True

    #flag = data[:,6]

    #print('flag', flag)

    global NNmodel
    NNmodel = model
    
    predictions_grad = model(inputs)

    v1 = torch.zeros_like(inputs, device = device)
    v2 = torch.zeros_like(inputs, device = device)
    v3 = torch.zeros_like(inputs, device = device)

    v1[:,0] = 1
    v2[:,1] = 1
    v3[:,2] = 1

    X = torch.split(g, 1, dim=1)
    x = X[0]
    y = X[1]
    t = X[2]

   
        
    predictions, du = torch.autograd.functional.vjp(prediction, (x, y, t), v1, create_graph=True)
    ux = du[0]
    uy = du[1]
    ut = du[2]

    #u_x_y = torch.cat((ux,uy), dim=1)

    u = predictions[:,0]
    v = predictions[:,1]
    p = predictions[:,2]

    u = torch.reshape(u, (u.shape[0],1))
    v = torch.reshape(v, (v.shape[0],1))
    p = torch.reshape(p, (p.shape[0],1))


    predictions1, dv = torch.autograd.functional.vjp(prediction, (x, y, t), v2, create_graph=True)
    vx = dv[0]
    vy = dv[1]
    vt = dv[2]

    #v_x_y = torch.cat((vx,vy), dim=1)

    predictions2, dp = torch.autograd.functional.vjp(prediction, (x, y, t), v3, create_graph=True)
    px = dp[0]
    py = dp[1]
    #pt = dp[2]

    v4 = torch.ones_like(x)
    v5 = torch.zeros_like(x)
    C, H_u = torch.autograd.functional.vhp(pred_hessian_u, (x,y,t), (v4,v4,v5), create_graph=True)

    C, H_v = torch.autograd.functional.vhp(pred_hessian_v, (x, y, t), (v4, v4, v5), create_graph=True)

    _, H_p = torch.autograd.functional.vhp(pred_hessian_p, (x,y,t), (v4,v4, v5), create_graph= True)

    u_xx = H_u[0]
    u_yy = H_u[1]
    v_xx = H_v[0]
    v_yy = H_v[1]
    p_xx = H_p[0]
    p_yy = H_p[1]

    #dv2_dx2 = torch.autograd.grad(vx, x, torch.ones(x.shape[0], 1).to(device), create_graph=True)
    #dv2_dy2 = torch.autograd.grad(vy, y, torch.ones(y.shape[0], 1).to(device), create_graph=True)
    '''
    u = predictions_grad[:,0]
    v = predictions_grad[:,1]
    p = predictions_grad[:,2]

    u = torch.reshape(u, (u.shape[0],1))
    v = torch.reshape(v, (v.shape[0],1))
    p = torch.reshape(p, (p.shape[0],1)) 
    
    u_x_y_t = torch.autograd.grad(predictions_grad , g, torch.ones([inputs.shape[0], 3]).to(device), retain_graph=True, create_graph=True)[0]

    print(u_x_y_t.shape)
    u_xx_yy_tt = torch.autograd.grad(u_x_y_t, g, torch.ones([inputs.shape[0],3]).to(device),
                create_graph=True)[0]

    v_x_y_t = torch.autograd.grad(v,g,torch.ones([inputs.shape[0], 1]).to(device),
                retain_graph=True, create_graph=True, allow_unused=True)[0]
    v_xx_yy_tt = torch.autograd.grad(v_x_y_t, g, torch.ones([inputs.shape[0],3]).to(device),
                create_graph=True)[0]

    p_x_y_t = torch.autograd.grad(p,g,torch.ones([inputs.shape[0], 1]).to(device),
                retain_graph=True, create_graph=True, allow_unused=True)[0]
    p_xx_yy_tt = torch.autograd.grad(p_x_y_t, g, torch.ones([inputs.shape[0],3]).to(device),
                create_graph=True)[0]
    
    print('uxx', u_xx_yy_tt.shape)

    u_x = u_x_y_t[:,[0]]
    u_y = u_x_y_t[:,[1]]
    u_t = u_x_y_t[:,[2]]
    u_xx = u_xx_yy_tt[:,[0]]
    u_yy = u_xx_yy_tt[:,[1]]

    v_x = v_x_y_t[:,[0]]
    v_y = v_x_y_t[:,[1]]
    v_t = v_x_y_t[:,[2]]
    v_xx = v_xx_yy_tt[:,[0]]
    v_yy = v_xx_yy_tt[:,[1]]

    p_x = p_x_y_t[:,[0]]
    p_y = p_x_y_t[:,[1]]
    p_t = p_x_y_t[:,[2]]
    p_xx = p_xx_yy_tt[:,[0]]
    p_yy = p_xx_yy_tt[:,[1]]
    ''' 
   
    continuity = ux + vy
    ns1 = ut + u*ux + v*uy + (1/rho)*px - nu*(u_xx + u_yy)
    ns2 = vt + u*vx + v*vy + (1/rho)*py - nu*(v_xx + v_yy)
        
    ps = p_xx + p_yy + rho*(ux*ux + 2*uy*vx + vy*vy) 

    
    target1 = torch.zeros_like(continuity, device=device)
    target2 = torch.zeros_like(ns1, device=device)
    target3 = torch.zeros_like(ns2, device=device)
    target4 = torch.zeros_like(ps, device=device)

    loss_continuity = loss_function(continuity, target1)
    loss_ns1 = loss_function(ns1, target2)
    loss_ns2 = loss_function(ns2, target3)
    loss_ps = loss_function(ps, target4)
    

    loss_variable = loss_function(predictions, exact)

    if epoch < 5000:
        #loss = loss_continuity + loss_ns1 + loss_ns2 + loss_ps + loss_variable
        loss = loss_continuity + loss_variable
    else:
        loss = loss_continuity + loss_ns1 + loss_ns2 + loss_ps + 0.1 * loss_variable

    if(batch == 0 and grank==0):
        if(epoch == 1):
            f = open("./result/loss.txt", "wb")

        else:
            f = open("./result/loss.txt", "a")
            #f.writelines([str(loss_continuity.item()) + " " + str(loss_ns1.item()) + " " + str(loss_ns2.item()) + " " + str(loss_variable.item()) + "\n"])
            f.writelines(['Epoch: {:03d}, LossContinuity: {:.9f}, LossNS1: {:.9f}, LossNS2: {:.9f}, LossPS: {:.9f}, LossPred: {:.9f}'.
            format(epoch, loss_continuity.item(), loss_ns1.item(), loss_ns2.item(), loss_ps.item(), loss_variable.item()) + '\n'])
            #f.writelines(['Epoch: {:03d}, LossContinuity: {:.9f}, LossNS1: {:.9f}, LossNS2: {:.9f}'.
            #format(epoch, loss_continuity.item(), loss_ns1.item(), loss_ns2.item()) + '\n'])

            f.close()

   
   
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
        print('DEBUG: args.benchrun:', args.benchrun)
        print('DEBUG: args.solution_int:', args.solution_int, '\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu',lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        # deterministic testrun
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

    train_data, test_data, X_in, V_p_star = h5_loader()


    u_min = V_p_star[:,0].min()
    u_max = V_p_star[:,0].max()
    v_min = V_p_star[:,1].min()
    v_max = V_p_star[:,1].max()
    p_min = V_p_star[:,2].min()
    p_max = V_p_star[:,2].max()

    rho = 1.2
    nu = 1.516e-5

    X_in = torch.from_numpy(X_in).float().to(device)

    train_len = len(train_data)
    test_len = len(test_data)

    # restricts data loading to a subset of the dataset exclusive to the current process
    args.shuff = args.shuff and not args.testrun
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset = GenerateDataset([x for x in range(train_len)]),
                            num_replicas=gwsize, rank=grank, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset = TestDataset([x for x in range(test_len)]),
                            num_replicas=gwsize, rank=grank, shuffle=True)

    # distribute dataset to workers
    # persistent workers
    pers_w = True if args.nworker>1 else False

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    train_loader = torch.utils.data.DataLoader(dataset = GenerateDataset([x for x in range(train_len)]), batch_size=args.batch_size,
                                               sampler = train_sampler,
                                               num_workers=args.nworker, pin_memory=False,
                                               persistent_workers=pers_w, drop_last=True,
                                               prefetch_factor=args.prefetch, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset = TestDataset([x for x in range(test_len)]), batch_size=2,
                                              sampler=test_sampler, num_workers=args.nworker, pin_memory=False,
                                              persistent_workers=pers_w, drop_last=True,
                                              prefetch_factor=args.prefetch, **kwargs)

    '''train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               pin_memory=True, drop_last=True,
                                               prefetch_factor=args.prefetch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2,
                                              pin_memory=True, drop_last=True,
                                              prefetch_factor=args.prefetch)'''

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n')

    # create model

    layers = np.array([3, 256, 256, 256, 256, 256, 3])
    model = Taylor_green_vortex_PINN(layers).to(device)
   
    print(device)
    # distribute model too workers
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model,\
                        device_ids = [device], output_device=device, find_unused_parameters = False)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters = False)

    '''if args.cuda:
        distrib_model = nn.parallel.DistributedDataParallel(model,\
            device_ids=[device], output_device=device)
    else:
        distrib_model = nn.parallel.DistributedDataParallel(model)'''

    # optimizer
    optimizerS = torch.optim.SGD(distrib_model.parameters(), lr=args.lr)
    optimizerA = torch.optim.Adam(distrib_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizerB = torch.optim.LBFGS(distrib_model.parameters(), lr=args.lr, max_iter=args.epochs, 
                 max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None) 
    optimizerL = Lion(model.parameters(), lr = args.lr)

    optimizer = optimizerA

    scheduler_lr1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler_lr2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)

    scheduler_lr = scheduler_lr2
    
    # resume state
    start_epoch = 1
    best_acc = np.Inf
    res_name = 'checkpoint_red_NC.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(program_dir + '/' + res_name , map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if grank == 0:
                print(f'WARNING: restarting from {start_epoch} epoch')
        except:
            if grank == 0:
                print(f'WARNING: restart file cannot be loaded, restarting!')
    if start_epoch >= args.epochs:
        if grank == 0:
            print(f'WARNING: given epochs are less than the one in the restart file!\n'
                  f'WARNING: SYS.EXIT is issued')
        dist.destroy_process_group()
        sys.exit()

    # start training/testing loop
    if grank==0:
        print('TIMER broadcast:', time.time()-st, 's')
        print(f'\nDEBUG: start training')
        print(f'------------------------------------------')

    et = time.time()
    loss_acc_list = []
    lr_graph = []
    for epoch in range(start_epoch, args.epochs + 1):

        lt = time.time()

        #training
        if args.benchrun and epoch==args.epochs:
            with torch.autograd.profiler.profile(use_cuda=args.cuda, profile_memory=True) as prof:
                loss_acc = train(distrib_model, device, train_loader, optimizer,epoch, grank, gwsize, rho, nu)
        else:
            loss_acc = train(distrib_model, device, train_loader, optimizer, epoch, grank, gwsize, rho, nu)

        #if grank == 0 and lrank == 0:
        loss_acc_list.append(loss_acc)

        # testing
        acc_test = test(distrib_model, device, test_loader, grank, gwsize, rho, nu, epoch)

        scheduler_lr.step()
        
        # current learning rate
        lr_cur = scheduler_lr.get_lr()
        
        lr_graph.append(lr_cur)

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
        is_best = loss_acc <= best_acc
        if epoch % args.restart_int == 0 and not args.benchrun:

            save_state(epoch, model, loss_acc, optimizer, res_name, grank, gwsize, is_best)
            #save_state(epoch, model, loss_acc, optimizer, res_name)
            best_acc = min(loss_acc, best_acc)
        if epoch % args.solution_int == 0:    
            V_p_pred_norm = distrib_model(X_in)
            u_pred, v_pred, p_pred = denormalize(V_p_pred_norm, u_min, u_max, v_min, v_max, p_min, p_max)
            result = [V_p_star,X_in,V_p_pred_norm, u_pred,v_pred, p_pred, loss_acc_list, epoch, lr_graph]
            f = open('./result/TGV/result_Taylor_green_vortex_reduced_NC_'+str(epoch)+'.pkl', 'wb')
            pickle.dump(result, f)
            f.close()

        if grank==0:
            print(epoch)

    #finalise training
    # save final state
    if not args.benchrun:
        save_state(epoch, model, loss_acc, optimizer, res_name, grank, gwsize, True)
        #save_state(epoch, model, loss_acc, optimizer, res_name)
    dist.barrier()


    if grank == 0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: training results:\n')
        print(f'TIMER: first epoch time: {first_ep_t} s')
        print(f'TIMER: last epoch time: {time.time() - lt} s')
        print(f'TIMER: total epoch time: {time.time() - et} s')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time() - et} s')


    V_p_pred_norm = distrib_model(X_in)
    u_pred, v_pred, p_pred = denormalize(V_p_pred_norm, u_min, u_max, v_min, v_max, p_min, p_max)

    if grank==0:
        f_time = time.time() - st
        print(f'TIMER: final time: {f_time} s')

    result = [V_p_star, X_in, V_p_pred_norm, u_pred, v_pred, p_pred, loss_acc_list, acc_test, lr_graph]
    f = open('./result/result_Taylor_green_vortex_reduced_NC_'+str(args.epochs)+'.pkl', 'wb')

    pickle.dump(result, f)
    f.close()

    # clean-up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    sys.exit()




