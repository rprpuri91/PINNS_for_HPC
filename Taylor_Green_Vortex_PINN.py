import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from torch import nn
import argparse
import sys, os, time, random, shutil
import torch.distributed as dist

torch.manual_seed(1234)

np.random.seed(1234)
class Preprocessing_Taylor_Green():
    def __init__(self, rho, nu, n):

        self.rho = rho
        self.nu = nu
        self.n = n

        x_values = np.arange(0, np.pi, 0.1).tolist()
        y_values = np.arange(0, np.pi, 0.1).tolist()
        t = np.arange(0, 1, 0.01)
        x_values.append(np.pi)
        y_values.append(np.pi)

        x,y = np.meshgrid(x_values, y_values)
        #y0,x0 = np.meshgrid(y_values, x_values)

        #X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

        x1, t1 = np.meshgrid(x, t)
        y1, t2 = np.meshgrid(y, t)
        #y2, t2 = np.meshgrid(y0,t)

        self.X_in = np.hstack([x1.flatten()[:,None], y1.flatten()[:,None], t1.flatten()[:,None]])
        X0 = np.zeros(y1.flatten().shape)
        Xpi = np.zeros(y1.flatten().shape)
        Xpi[:] = np.pi


        self.V_star = self.velocity(self.X_in)
        self.p_star = self.pressure(self.X_in)

        self.V_min = self.V_star.min()
        self.V_max = self.V_star.max()
        self.p_min = self.p_star.min()
        self.p_max = self.p_star.max()

        self.X_bottom = np.vstack([x1.flatten(), X0, t1.flatten()]).T
        self.X_top = np.vstack([x1.flatten(), Xpi, t1.flatten()]).T
        self.X_left = np.vstack([X0, x1.flatten(), t1.flatten()]).T
        self.X_right = np.vstack([Xpi, x1.flatten(), t1.flatten()]).T

        self.l1 = len(x.flatten())

        X = self.X_in[:self.l1]
        V = self.V_star[:self.l1]

        plt.quiver(X[:, 0], X[:, 1], V[:, 0], V[:, 1])
        plt.show()


    def velocity(self, X):
        #F(t) = e^(-2*nu*t)
        #u = sinx*cosy*F(t)
        #v = -cosx*siny*F(t)

        U = []


        for i in range(len(X)):
            f = -2*self.nu*X[i][2]
            F = np.exp(f)
            u0 = np.sin(X[i][0])*np.cos(X[i][1])*F
            v0 = -np.cos(X[i][0])*np.sin(X[i][1])*F
            U0 = [u0,v0]
            U.append(U0)

        U_numpy = np.array(U)

        return U_numpy

    def pressure(self, X):
        #p = (rho/4)*(cos2x + sin2y)*F^2

        p = []

        for i in range(len(X)):
            f = -2 * self.nu * X[i][2]
            F = np.exp(f)
            p0 = self.rho*(np.cos(2*X[i][0]) + np.sin(2*X[i][1]))*F
            p.append(p0)

        p_numpy = np.array(p)

        return p_numpy

    def normalize(self, V, p):
        V_norm = (V - self.V_min) / (self.V_max - self.V_min)

        p_norm = (p -self.p_min) / (self.p_max - self.p_min)
        return V_norm, p_norm


    def train_test(self,X):
        N = int(self.n * len(X))

        idx = np.random.choice(X.shape[0], N, replace=False)

        X_train = X[idx,:]

        X_test = np.delete(X, idx, axis=0)

        V_train = self.velocity(X_train)
        V_test = self.velocity(X_test)

        p_train = self.pressure(X_train)
        p_test = self.pressure(X_test)

        V_train_norm, p_train_norm = self.normalize(V_train,p_train)

        V_test_norm, p_test_norm = self.normalize(V_test, p_test)

        V_p_train = np.vstack([V_train_norm[:,0], V_train_norm[:,1], p_train_norm])
        V_p_test = np.vstack([V_test_norm[:,0], V_test_norm[:,1], p_test_norm])

        return V_p_train.T, X_train, V_p_test.T, X_test

    def data_generation(self):

        N2 = int(0.3 * len(self.X_in))

        idx = np.random.choice(self.X_in.shape[0], N2, replace=False)
        X_domain = self.X_in[idx, :]

        V_p_train_domain, X_train_domain, V_p_test_domain, X_test_domain = self.train_test(X_domain)
        V_p_train_left, X_train_left, V_p_test_left, X_test_left = self.train_test(self.X_left)
        V_p_train_right, X_train_right, V_p_test_right, X_test_right = self.train_test(self.X_right)
        V_p_train_top, X_train_top, V_p_test_top, X_test_top = self.train_test(self.X_top)
        V_p_train_bottom, X_train_bottom, V_p_test_bottom, X_test_bottom = self.train_test(self.X_bottom)

        h5 = h5py.File('data_Taylor_Green_Vortex.h5', 'w')
        g1 = h5.create_group('domain')
        g1.create_dataset('data1', data=X_train_domain)
        g1.create_dataset('data2', data=V_p_train_domain)
        g1.create_dataset('data3', data=X_test_domain)
        g1.create_dataset('data4', data=V_p_test_domain)

        g2 = h5.create_group('left')
        g2.create_dataset('data1', data=X_train_left)
        g2.create_dataset('data2', data=V_p_train_left)
        g2.create_dataset('data3', data=X_test_left)
        g2.create_dataset('data4', data=V_p_test_left)

        g3 = h5.create_group('right')
        g3.create_dataset('data1', data=X_train_right)
        g3.create_dataset('data2', data=V_p_train_right)
        g3.create_dataset('data3', data=X_test_right)
        g3.create_dataset('data4', data=V_p_test_right)

        g4 = h5.create_group('top')
        g4.create_dataset('data1', data=X_train_top)
        g4.create_dataset('data2', data=V_p_train_top)
        g4.create_dataset('data3', data=X_test_top)
        g4.create_dataset('data4', data=V_p_test_top)

        g5 = h5.create_group('bottom')
        g5.create_dataset('data1', data=X_train_bottom)
        g5.create_dataset('data2', data=V_p_train_bottom)
        g5.create_dataset('data3', data=X_test_bottom)
        g5.create_dataset('data4', data=V_p_test_bottom)

        g6 = h5.create_group('full')
        g6.create_dataset('data1', data=self.X_in)
        g6.create_dataset('data2', data=self.V_star)
        g6.create_dataset('data3', data=self.p_star)

        h5.close()

def h5_loader(path):
    h5 = h5py.File(path, 'r')

    try:
        domain = h5.get('domain')
        left = h5.get('left')
        right = h5.get('right')
        top = h5.get('top')
        bottom = h5.get('bottom')


        X_train_domain = np.array(domain.get('data1'))
        V_p_train_domain = np.array(domain.get('data2'))
        X_test_domain = np.array(domain.get('data3'))
        V_p_test_domain = np.array(domain.get('data4'))

        X_train_left = np.array(left.get('data1'))
        V_p_train_left = np.array(left.get('data2'))
        X_test_left = np.array(left.get('data3'))
        V_p_test_left = np.array(left.get('data4'))

        X_train_right = np.array(right.get('data1'))
        V_p_train_right = np.array(right.get('data2'))
        X_test_right = np.array(right.get('data3'))
        V_p_test_right = np.array(right.get('data4'))

        X_train_top = np.array(top.get('data1'))
        V_p_train_top = np.array(top.get('data2'))
        X_test_top = np.array(top.get('data3'))
        V_p_test_top = np.array(top.get('data4'))

        X_train_bottom = np.array(bottom.get('data1'))
        V_p_train_bottom = np.array(bottom.get('data2'))
        X_test_bottom = np.array(bottom.get('data3'))
        V_p_test_bottom = np.array(bottom.get('data4'))

        print(X_train_domain.shape)
        print(X_train_left.shape)
        print(X_train_right.shape)
        print(X_train_top.shape)
        print(X_train_bottom.shape)

        X_train = np.vstack([X_train_domain, X_train_left, X_train_right, X_train_top, X_train_bottom])
        X_test = np.vstack([X_test_domain, X_test_left, X_test_right, X_test_top, X_test_bottom])
        V_p_train = np.vstack([V_p_train_domain, V_p_train_left, V_p_train_right, V_p_train_left, V_p_train_top,V_p_train_bottom])
        V_p_test = np.vstack([V_p_test_domain, V_p_test_left, V_p_test_right, V_p_test_left, V_p_test_top, V_p_test_bottom])

    except Exception as e:
        print(e)

    return X_train, V_p_train, X_test, V_p_test

def pars_ini():
    global args

    parser = argparse.ArgumentParser(description='PyTorch actuated')

    #IO
    parser.add_argument('--data-dir', default='./', help='location of the training dataset')
    parser.add_argument('--restart-int', type=int, default=10, help='restart interval per epoch (default: 10)')

    #model
    parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs (default: 10)')
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
    parser.add_argument('--log-int', type=int, default=10,
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

        # loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        self.layers = layers

        # layers
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self,X):
        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X)
        X = X.to(self.device)

        x = self.scaling(X)
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

            a = self.activation2(z)

        a = self.activation(a)

        a = self.linears[-1](a)

        return a

    def scaling(self, X):

        mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
        # preprocessing input
        x = (X - mean) / (std)  # feature scaling

        return x

# save state of the training
def save_state(epoch,distrib_model,loss_acc,optimizer,res_name,grank,gwsize,is_best):
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
            print(f'DEBUG: state in {grank} is saved on epoch:{epoch} in {time.time()-rt} s')

# deterministic dataloader
'''def seed_worker(worker_id):
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
    return res'''

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
    predictions = model.forward(g)
    return predictions

def loss(data, device):

    loss_function = nn.MSELoss()
    inputs = data[:-1][0]

    v1 = torch.zeros_like(inputs, device = device)
    v2 = torch.zeros_like(inputs, device = device)
    v3 = torch.zeros_like(inputs, device = device)

    v1[:,0] = 1
    v2[:,1] = 1
    v3[:,2] = 1

    X = torch.split(inputs, 1, dim=1)
    x = [0]
    y = [1]
    t = [2]

    predictions, du = torch.autograd.functional.vjp(prediction, (x, y, t), v1, create_graph=True)
    ux = du[0]
    uy = du[1]
    ut = du[2]

    u = predictions[:,0]
    v = predictions[:,1]
    t = predictions[:,2]

    predictions1, dv = torch.autograd.functional.vjp(prediction, (x, y, t), v2, create_graph=True)
    vx = dv[0]
    vy = dv[1]
    vt = dv[2]

    predictions2, dp = torch.autograd.functional.vjp(prediction, (x, y, t), v3, create_graph=True)
    px = dp[0]
    py = dp[1]
    pt = dp[2]

    du2_dx2 = torch.autograd.grad(ux, x, torch.ones(x.shape[0], 1).to(device))
    du2_dy2 = torch.autograd.grad(uy, y, torch.ones(y.shape[0],1).to(device))

    dv2_dx2 = torch.autograd.grad(vx, x, torch.ones(x.shape[0], 1).to(device))
    dv2_dy2 = torch.autograd.grad(vy, y, torch.ones(y.shape[0], 1).to(device))

    continuity = ux + vy
    ns1 = ut + u*ux + v*uy + (1/rho)*px - nu*(du2_dx2 + du2_dy2)
    ns2 = vt + u*vx + v*vy + (1/rho)*py - nu*(dv2_dx2 + dv2_dy2)

    target1 = torch.zeros_like(continuity, device=device)
    target2 = torch.zeros_like(ns1, device=device)
    target3 = torch.zeros_like(ns2, device=device)

    loss_continuity = loss_function(continuity, target1)
    loss_ns1 = loss_function(ns1, target2)
    loss_ns2 = loss_function(ns2, target3)

    return loss_continuity + loss_ns1 + loss_ns2


def main():

    # get parse arguments
    pars_ini()

    # check for CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # get directory
    program_dir = os.getcwd()

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
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
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

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu', lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

    X_train, V_p_train, X_test, V_p_test = h5_loader(args.data_dir)

    # create dataset
    train_dataset = torch.utils.data.TensorDataset(X_train,V_p_train)
    test_dataset = torch.utils.data.TensorDataset(X_test,V_p_test)

    # distribute dataset to workers
    # persistent workers
    #pers_w = True if args.nworker>1 else False

    #kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    '''train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=args.batch_size,
                                               num_worker=args.nworker, pin_memory=True,
                                               persistent_workers=pers_w, drop_last=True,
                                               prefetch_fsactor=args.prefetch, **kwargs)
    test_loader = torch.utils.data.Dataloader(test_dataset, batch_size=2,
                                              num_worker=args.nworker, pin_memory=True,
                                              persistent_workers=pers_w, drop_last=True,
                                              prefetch_fsactor=args.prefetch, **kwargs)'''

    train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=args.batch_size,
                                               pin_memory=True, drop_last=True,
                                               prefetch_fsactor=args.prefetch)
    test_loader = torch.utils.data.Dataloader(test_dataset, batch_size=2,
                                              pin_memory=True, drop_last=True,
                                              prefetch_fsactor=args.prefetch)

    if grank==0:
        print(f'TIMER: read data: {time.time()-st} s\n')

    # create model
    layers = np.array([3, 60, 60, 60, 60, 60, 3])
    global model
    model = Taylor_green_vortex_PINN(layers).to(device)

    # distribute model tpo workers
    '''global distrib_model
    if args.cuda:
        distrib_model = torch.nn.parellel.DistributedDataParallel(model,\
                        device_ids = [device], output_device=device)
    else:
        distrib_model = torch.nn.parallel.DistributedDataParallel(model)'''

    # optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # resume state
    start_epoch = 0
    best_acc = np.Inf
    res_name = 'checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {'cpu:%d' % 0: 'cpu:%d' % lrank}
            checkpoint = torch.load(program_dir + '/' + res_name, map_location=loc)
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

    et = time.time()
    loss_acc_list = []

    for epoch in range(start_epoch, args.epochs):
        lt = time.time()
        loss_acc = 0.0
        count = 0
        for data in train_loader:
            print(data)
            inputs = data[:-1][0]

            optimizer.zero_grad()
            #predictions = distrib_model(inputs)

            loss = loss(data, device)

            loss.backward()
            optimizer.step()
            loss_acc+= loss.item()
            if count % args.log_int == 0 and grank == 0 and lrank == 0:
                print(f'Epoch: {epoch} / {100 * (count + 1) / len(train_loader):.2f}% complete', \
                      f' / {time.time() - lt:.2f} s / accumulated loss: {loss_acc}\n')
            count += 1
        if grank == 0 and lrank == 0:
            loss_acc_list.append(loss_acc)

        # if a better state is found
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0 and not args.benchrun:
            save_state(epoch, model, loss_acc, optimizer, res_name, grank, gwsize, is_best)

            # reset best_acc
            best_acc = min(loss_acc, best_acc)

        # profiling
        if grank == 0:
            print('TIMER: epoch time:', time.time()-lt, 's')

        if epoch == start_epoch:
            first_ep_t = time.time() - lt

    #finalise training
    # save final state
    if not args.benchrun:
        save_state(epoch, model, loss_acc, optimizer, res_name, grank, gwsize, True)
    dist.barrier()

    if grank == 0:

        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: training results:')
        print(f'TIMER: first epoch time: {first_ep_t} s')
        print(f'TIMER: last epoch time: {time.time() - lt} s')
        print(f'TIMER: total epoch time: {time.time() - et} s')
        print(f'TIMER: average epoch time: {(time.time() - et) / args.epochs} s')
        if epoch > 0:
            print(f'TIMER: total epoch-1 time: {time.time() - et - first_ep_t} s')
            print(f'TIMER: average epoch-1 time: {(time.time() - et - first_ep_t) / (args.epochs - 1)} s')
        print('DEBUG: memory req:', int(torch.cuda.memory_reserved(lrank) / 1024 / 1024), 'MB') \
            if args.cuda else 'DEBUG: memory req: - MB'

        torch.save(loss_acc_list, './loss_acc_per_ep.pt')

# testing loop
    et = time.time()
    #model.eval()
    test_loss = 0.0
    #mean_sqr_diff = []
    #count = 0
    with torch.no_grad():
        for data in test_loader:
            print(data)
            inputs = data[:-1][0]
            print(inputs)
            #predictions = distrib_model(inputs)

            loss = loss(data, device)

            test_loss+= loss.item()/inputs.shape[0]

            #count+=1
    if grank == 0:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: testing results:')
        print(f'TIMER: total testing time: {time.time() - et} s')

    # finalise testing
    # mean from gpus

    if grank==0:
        print(f'TIMER: final time: {time.time() - st} s')

        # clean-up
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
    sys.exit()


path = 'data_Taylor_Green_Vortex.h5'

h5_loader(path)

rho = 1.2

nu = 1.516e-5

n = 0.8

preprocessing = Preprocessing_Taylor_Green(rho,nu,n)

#preprocessing.data_generation()


