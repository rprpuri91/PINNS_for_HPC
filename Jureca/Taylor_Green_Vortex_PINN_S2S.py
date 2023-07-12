import numpy as np
# import torch
#from deepxde.backend import torch
import torch
import h5py
from torch import nn
import argparse
import pandas as pd
import sys, os, time, random, shutil
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader

import deepxde as dde


def pars_ini():
    global args

    parser = argparse.ArgumentParser(description='PyTorch actuated')

    # IO
    parser.add_argument('--data-dir', default='./', help='location of the training dataset')
    parser.add_argument('--restart-int', type=int, default=10, help='restart interval per epoch (default: 10)')

    # model
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
    parser.add_argument('--solution-int', type=int, default=50,
                        help='saving prediction result after interval (default: 50)')

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

    def forward(self, X):
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

        # print("\tIn Model: input size", X.size(), "output size", a.size())

        return a


def scaling(X):
    mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
    # preprocessing input
    x = (X - mean) / (std)  # feature scaling

    return x


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

        # print(V_p_star)

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

        train_data = np.vstack([train_domain, train_left, train_right, train_top, train_bottom])
        test_data = np.vstack([test_domain, test_left, test_right, test_top, test_bottom])

        print('train',train_data.shape)
        print('test', test_data.shape)
        '''print('########################################')
        for i in range(len(train_data)):
            print(train_data[i])
            print("\n")
        print('#######################################')
        '''
    except Exception as e:
        print(e)

    return train_data, test_data, X_in, V_p_in

class Train_dataset(Dataset):
    def __init__(self, root):
        self.train_data,_,_,_ = h5_loader(root)
        self.len = len(self.train_data)

        def __getitem__(self, index):
            data0 = self.train_data[index]
            data = torch.from_numpy(data0).float()
            return data

        def __len__(self):
            return self.len

class Test_dataset(Dataset):
    def __init__(self, root):
        _,self.test_data,_,_ = h5_loader(root)
        self.len = len(self.test_data)

        def __getitem__(self, index):
            data0 = self.test_data[index]
            data = torch.from_numpy(data0).float()
            return data

        def __len__(self):
            return self.len

def scaling(X):

    mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
    # preprocessing input
    x = (X - mean) / (std)  # feature scaling

    return x

def save_state(t,epoch,distrib_model,loss_acc,optimizer,res_name, grank, gwsize, is_best):#,grank,gwsize,is_best):
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
                'optimizer' : optimizer.state_dict(),
                'time_step' : t}

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

def total_loss(model, data, device, rho, nu):

    loss_function = nn.MSELoss()

    inputs = data[0]

    g = inputs.clone()
    g.requires_grad = True

    exact = data[1]

    outputs = model(inputs)

    u = outputs[:, 0]
    v = outputs[:, 1]
    p = outputs[:, 2]

    ux = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    uy = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    ut = dde.grad.jacobian(outputs, inputs, i=0, j=2)

    vx = dde.grad.jacobian(outputs, inputs, i=1, j=0)
    vy = dde.grad.jacobian(outputs, inputs, i=1, j=1)
    vt = dde.grad.jacobian(outputs, inputs, i=1, j=2)

    px = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    py = dde.grad.jacobian(outputs, inputs, i=2, j=1)

    u_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)

    v_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)

    #p_xx = dde.grad.hessian(outputs, inputs, component=2, i=0, j=0)
    #p_yy = dde.grad.hessian(outputs, inputs, component=2, i=1, j=1)


    continuity = ux + vy
    ns1 = ut + u*ux + v*uy + (1/rho)*px - nu*(u_xx + u_yy)
    ns2 = vt + u*vx + v*vy + (1/rho)*py - nu*(v_xx + v_yy)

    target1 = torch.zeros_like(continuity, device=device)
    target2 = torch.zeros_like(ns1, device=device)
    target3 = torch.zeros_like(ns2, device=device)

    loss_continuity = loss_function(continuity, target1)
    loss_ns1 = loss_function(ns1, target2)
    loss_ns2 = loss_function(ns2, target3)

    loss_variable = loss_function(outputs, exact)

    return loss_continuity + loss_ns1 + loss_ns2 + loss_variable

def denormalize(self, V_p_norm, u_min, u_max, v_min, v_max, p_min, p_max):
    u_norm = V_p_norm[:,0]
    v_norm = V_p_norm[:,1]
    p_norm = V_p_norm[:,2]
    u = ((u_norm + 1) * (u_max - u_min) / 2) + u_min
    v = ((v_norm + 1) * (v_max - v_min) / 2) + v_min
    p = ((p_norm + 1) * (p_max - p_min) / 2) + p_min

    return u_norm, v_norm, p_norm

def main():
    ts_df = pd.read_csv('../data/timeSteps.csv', delimiter=",", header=None)
    timeSteps= np.array(ts_df.values)

    # set default tensor generation device
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

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
    lwsize = torch.cuda.device_count() if args.cuda else 0  # local world size - per node
    gwsize = dist.get_world_size()  # global world size - per run
    grank = dist.get_rank()  # global rank - assign per run
    lrank = dist.get_rank() % lwsize  # local rank - assign per node

    if grank == 0:
        print('TIMER: initialise:', time.time() - st, 's')
        # print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
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
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu', lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        # deterministic testrun
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

    rho = 1
    nu = 0.1

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n')

    # create model

    layers = np.array([3, 256, 256, 256, 256, 256, 3])
    model = Taylor_green_vortex_PINN(layers).to(device)

    print(device)
    # distribute model too workers
    if args.cuda:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model, \
                                                                  device_ids=[device], output_device=device,
                                                                  find_unused_parameters=False)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    current_epoch = 1
    best_acc = np.Inf
    t=0
    res_name = 'checkpoint_red_initial.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(program_dir + '/' + res_name , map_location=loc)
            current_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            time_step = checkpoint['time_step']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if grank == 0:
                print(f'WARNING: restarting from {current_epoch} epoch')
        except:
            if grank == 0:
                print(f'WARNING: restart file cannot be loaded, restarting!')
    if current_epoch >= args.epochs:
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

    train_additional=[]
    ts_range = []

    et = time.time()
    test_acc = []
    train_loss = []

    for t in range(t,timeSteps):
        ts1 = str(int(timeSteps[t][0]))
        #print(ts1)
        if(ts1=="1"):
            path = '../data/S2S/data_Taylor_Green_Vortex_reduced_01.h5'
        else:
            path = '../data/S2S/data_Taylor_Green_Vortex_reduced_'+ts1+'.h5'

        train_data, test_data, X_in, V_p_in = h5_loader(path)

        train_len = len(train_data)
        test_len = len(test_data)

        # restricts data loading to a subset of the dataset exclusive to the current process
        args.shuff = args.shuff and not args.testrun
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=Train_dataset([x for x in range(train_len)], root=path),
            num_replicas=gwsize, rank=grank, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=Test_dataset([x for x in range(test_len)], root=path),
            num_replicas=gwsize, rank=grank, shuffle=True)

        # distribute dataset to workers
        # persistent workers
        pers_w = True if args.nworker > 1 else False

        # deterministic testrun - the same dataset each run
        kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

        train_loader = torch.utils.data.DataLoader(dataset=Train_dataset([x for x in range(train_len)], root=path),
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=args.nworker, pin_memory=False,
                                                   persistent_workers=pers_w, drop_last=True,
                                                   generator=torch.Generator(device=device),
                                                   prefetch_factor=args.prefetch, **kwargs)

        test_loader = torch.utils.data.DataLoader(dataset=Test_dataset([x for x in range(test_len)], root=path), batch_size=2,
                                                  sampler=test_sampler, num_workers=args.nworker, pin_memory=False,
                                                  persistent_workers=pers_w, drop_last=True,
                                                  generator=torch.Generator(device=device),
                                                  prefetch_factor=args.prefetch, **kwargs)

        X_in_test = torch.from_numpy(test_data[:,0:3]).to(device)
        V_p_testing = []
        for epoch in range(current_epoch, args.epochs + 1):

            lt = time.time()

            # training
            if args.benchrun and epoch == args.epochs:
                with torch.autograd.profiler.profile(use_cuda=args.cuda, profile_memory=True) as prof:
                    loss_acc = train(distrib_model, device, train_loader, optimizer, epoch, grank, gwsize, rho, nu)
            else:
                loss_acc = train(distrib_model, device, train_loader, optimizer, epoch, grank, gwsize, rho, nu)

            # if grank == 0 and lrank == 0:
            # testing
            acc_test = test(distrib_model, device, test_loader, grank, gwsize, rho, nu, epoch)

            test_acc.append(acc_test)
            train_loss.append(loss_acc)
            # scheduler_lr.step()

            # current learning rate
            #lr_cur = scheduler_lr.get_lr()

            #lr_graph.append(lr_cur)

            if epoch == current_epoch:
                first_ep_t = time.time() - lt

            # final epoch
            if epoch + 1 == args.epochs:
                train_loader.last_epoch = True
                test_loader.last_epoch = True

            if grank == 0:
                print(f'\n--------------------------------------------------------')
                print(f'TIMER: epoch time:', time.time() - lt, 's')
                print(f'DEBUG: accuracy:', acc_test, '%')
                if args.benchrun and epoch == args.epoch:
                    print(f'\n----------------------------------------')
                    print(f'DEBUG: benchmark of last epoch:\n')
                    what1 = 'cuda' if args.cuda else 'cpu'
                    print(prof.key_averages().table(sort_by='self_' + str(what1) + '_time_total'))

            # if a better state is found
            is_best = loss_acc <= best_acc
            if epoch % args.restart_int == 0 and not args.benchrun:
                save_state(t,epoch, model, loss_acc, optimizer, res_name, grank, gwsize, is_best)
                # save_state(epoch, model, loss_acc, optimizer, res_name)
                best_acc = min(loss_acc, best_acc)
            if epoch % args.solution_int == 0:
                V_p_pred_norm = distrib_model(X_in)
                u_pred, v_pred, p_pred = denormalize(V_p_pred_norm, u_min, u_max, v_min, v_max, p_min, p_max)
                result = [V_p_star, X_in, V_p_pred_norm, u_pred, v_pred, p_pred, loss_acc_list, epoch, lr_graph]
                f = open('./result/TGV/result_Taylor_green_vortex_reduced_initial01_' + str(epoch) + '.pkl', 'wb')
                pickle.dump(result, f)
                f.close()

            if grank == 0:
                print(epoch)

            V_p_test = distrib_model(X_in_test)
            V_p_testing.append(V_p_test)

            condition = torch.linalg.norm((V_p_test - V_p_testing[epoch-1]), 2) / torch.linalg.norm(V_p_test,2)

            if epoch > 1 and condition < 1e-8:
                test_TS = torch.hstack([X_in_test, V_p_test])
                train_additional.append(test_TS)
                break

    if not args.benchrun:
        save_state(epoch, model, loss_acc, optimizer, res_name, grank, gwsize, True)
        # save_state(epoch, model, loss_acc, optimizer, res_name)
    dist.barrier()

    if grank == 0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: training results:\n')
        print(f'TIMER: first epoch time: {first_ep_t} s')
        print(f'TIMER: last epoch time: {time.time() - lt} s')
        print(f'TIMER: total epoch time: {time.time() - et} s')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time() - et} s')

    '''V_p_pred_norm = distrib_model(X_in)
    u_pred, v_pred, p_pred = denormalize(V_p_pred_norm, u_min, u_max, v_min, v_max, p_min, p_max)'''

    if grank == 0:
        f_time = time.time() - st
        print(f'TIMER: final time: {f_time} s')

    '''result = [V_p_star, X_in, V_p_pred_norm, u_pred, v_pred, p_pred, loss_acc_list, acc_test, f_time]
    f = open('./result/result_Taylor_green_vortex_reduced.pkl', 'wb')

    pickle.dump(result, f)
    f.close()'''

    # clean-up
    dist.destroy_process_group()

'''if __name__ == "__main__":
    main()
    sys.exit()'''






