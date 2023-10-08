import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import numpy as np
import torch
import h5py
import re
#from sklearn.preprocessing import MinMaxScaler
import sys, os, time, random, shutil
import tikzplotlib as tkz
'''matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'Times New Roman',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "pgf.texsystem" : "xelatex"
})'''

plt.rcParams["font.family"]="Times New Roman"
matplotlib.rcParams.update({'font.size': 14})


torch.manual_seed(1234)

np.random.seed(1234)
class Preprocessing_Taylor_Green():
    def __init__(self, rho, nu, n):

        self.rho = rho
        self.nu = nu
        self.n = n

        x_values = np.arange(-np.pi, np.pi, 0.1).tolist()
        y_values = np.arange(-np.pi, np.pi, 0.1).tolist()
        t = np.arange(0, 100, 1)
        x_values.append(np.pi)
        y_values.append(np.pi)
        print(len(x_values))
        self.x_d = x_values[1:len(x_values)-1]
        self.y_d = y_values[1:len(y_values)-1]

        self.x_domain,self.y_domain = np.meshgrid(self.x_d,self.y_d)

        x,y = np.meshgrid(x_values, y_values)
        #y0,x0 = np.meshgrid(y_values, x_values)

        #X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])

        x1, t1 = np.meshgrid(x, t)
        y1, t2 = np.meshgrid(y, t)
        #y2, t2 = np.meshgrid(y0,t)

        self.X_in = np.hstack([x1.flatten()[:,None], y1.flatten()[:,None], t1.flatten()[:,None]])
        X0 = np.zeros(y1.flatten().shape)
        Xpi = np.zeros(y.flatten().shape)
        Xpi[:] = np.pi
        t0 = np.zeros((x.flatten().shape))
        t01 = np.zeros((x.flatten().shape))
        t01[:] = t[1]
        center = np.zeros(t.flatten().shape)
        center[:] = np.pi/2


        self.u_star, self.v_star = self.velocity(self.X_in)
        self.p_star = self.pressure(self.X_in)

        self.u_min = self.u_star.min()
        self.u_max = self.u_star.max()
        self.v_min = self.v_star.min()
        self.v_max = self.v_star.max()
        self.p_min = self.p_star.min()
        self.p_max = self.p_star.max()

        self.l1 = len(x.flatten())

        self.X_initial = np.vstack([x.flatten(), y.flatten(), t0]).T
        self.X_1 = np.vstack([x.flatten(), y.flatten(), t01]).T
        self.X_initial_01 = self.X_in[0:self.l1]
        self.X_bottom = np.vstack([x[0,:].flatten(), y[0,:].flatten()]).T
        self.X_top = np.vstack([x[0,:].flatten(), y[-1,:].flatten()]).T
        self.X_left = np.vstack([x[:,0].flatten(), y[:,0].flatten()]).T
        self.X_right = np.vstack([x[:,-1].flatten(), y[:,0].flatten()]).T
        self.X_center = np.vstack([center.flatten(), center.flatten(), t.flatten()]).T

        self.u, self.v = self.velocity(self.X_initial)
        self.p = self.pressure(self.X_initial)

        #self.fig, self.ax = plt.subplots(1,1)
        #self.animate()
        #plt.show()
        #self.ax.add_patch(patch.Rectangle((-np.pi, -np.pi),2*np.pi, 2*np.pi, fill=False))

        #self.ax.scatter(self.X_right[:,0], self.X_right[:,1])
        #c = ax.tricontourf(self.X_initial[:,0],self.X_initial[:,1],self.u, levels=7)
        ##fig.colorbar(c,ax=ax)
        #plt.show()

        print('Xleft',self.X_left.shape)
        print('X_initial',self.X_initial.shape)
        '''print(self.u_center)
        print(self.v_center)'''

        X = self.X_in[:self.l1]
        u = self.u_star[:self.l1]
        v = self.v_star[:self.l1]



        '''plt.quiver(self.X_center[:, 0], self.X_center[:, 1], self.u_center, self.v_center)
        plt.show()'''

    def animate(self):

        for t in range(0,100):
            print(t)
            X = self.X_in[t*self.l1:(t+1)*self.l1]
            u, v = self.velocity(X)
            self.ax.clear()
            c = self.ax.tricontourf(X[:, 0], X[:, 1], u, levels = 7)
            self.fig.colorbar(c, ax=self.ax)
            time.sleep(1)
            plt.show()

    def velocity(self, X):
        #F(t) = e^(-2*nu*t)
        #u = sinx*cosy*F(t)
        #v = -cosx*siny*F(t)

        u = []
        v = []

        for i in range(len(X)):
            f = -2*self.nu*X[i][2]
            F = np.exp(f)
            u0 = np.cos(X[i][0])*np.sin(X[i][1])*F
            v0 = -np.sin(X[i][0])*np.cos(X[i][1])*F
            #U0 = [u0,v0]
            u.append(u0)
            v.append(v0)

        u_numpy = np.array(u)
        v_numpy = np.array(v)

        return u_numpy, v_numpy

    def pressure(self, X):
        #p = (rho/4)*(cos2x + sin2y)*F^2

        p = []

        for i in range(len(X)):
            f = -4 * self.nu * X[i][2]
            F = np.exp(f)
            p0 = -(self.rho/4)*(np.cos(2*X[i][0]) + np.cos(2*X[i][1]))*F
            p.append(p0)

        p_numpy = np.array(p)

        return p_numpy

    def normalize01(self, u,v,p):
        u_norm = (u - self.u_min) / (self.u_max - self.u_min)
        v_norm = (v - self.v_min) / (self.v_max - self.v_min)
        p_norm = (p - self.u_min) / (self.u_max - self.u_min)
        return u_norm, v_norm, p_norm

    def normalize(self, u,v, p):

        u_norm = -1 + 2*((u - self.u_min) / (self.u_max - self.u_min))
        v_norm = -1 + 2*((v - self.v_min) / (self.v_max - self.v_min))
        p_norm = -1 + 2*((p - self.p_min) / (self.p_max - self.p_min))
        return u, v, p_norm


    def denormalize(self, u_norm, v_norm, p_norm):
        u = (((u_norm + 1) * (self.u_max - self.u_min)) / 2 ) + self.u_min
        v = (((v_norm + 1) * (self.v_max - self.v_min)) / 2) + self.v_min
        scale = (self.u_max + self.v_max) / (2 * self.p_max)
        p = (((scale * p_norm + 1) * (self.p_max - self.p_min)) / 2) + self.p_min
        return u, v, p

    def train_test(self,X, flag):
        N = int(self.n * len(X))

        idx = np.random.choice(X.shape[0], N, replace=False)

        idx2a=[]
        idx2b=[]

        for i in range(0, len(X)):
            if i%10 == 0:
                idx2b.append(i)
            else:
                idx2a.append(i)

        X_train = X[idx,:]

        X_test = np.delete(X, idx, axis=0)

        X_train1 = X[idx2a]
        X_test1 = X[idx2b]

        u_train, v_train = self.velocity(X_train1)
        u_test, v_test = self.velocity(X_test1)

        p_train = self.pressure(X_train1)
        p_test = self.pressure(X_test1)

        #u_train, v_train, p_train = self.normalize(u_train, v_train,p_train)

        #u_test, v_test, p_test = self.normalize(u_test, v_test, p_test)

        '''if flag == "domain":
            flag_mark = 1
        elif flag == "left" or flag == "right" or flag == "top" or flag == "bottom":
            flag_mark = 2
        elif flag == "initial":
            flag_mark = 0

        flag_train = torch.full((u_train.shape), flag_mark)
        flag_test = torch.full((u_test.shape), flag_mark)'''

        V_p_train = np.vstack([u_train, v_train, p_train])
        V_p_test = np.vstack([u_test, v_test, p_test])


        return V_p_train.T, X_train1, V_p_test.T, X_test1

    def X_gen(self,t):

        t0 = np.zeros((self.x_domain.flatten().shape))
        t1 = np.zeros((self.X_left.shape[0]))
        t2 = np.zeros((self.X_right.shape[0]))
        t3 = np.zeros((self.X_top.shape[0]))
        t4 = np.zeros((self.X_bottom.shape[0]))
        t0[:] = t
        t1[:] = t
        t2[:] = t
        t3[:] = t
        t4[:] = t
        X_domain = np.vstack([self.x_domain.flatten(), self.y_domain.flatten(), t0]).T
        X_left = np.vstack([self.X_left[:,0], self.X_left[:,1], t1]).T
        X_right = np.vstack([self.X_right[:,0], self.X_right[:,1], t2]).T
        X_top = np.vstack([self.X_top[:,0], self.X_top[:,1], t3]).T
        X_bottom = np.vstack([self.X_bottom[:,0], self.X_bottom[:,1], t4]).T

        return X_domain, X_left, X_right, X_top, X_bottom

    def data_generation(self):

        t=20

        X_in1, X_left, X_right, X_top, X_bottom = self.X_gen(t)

        print("size", X_in1.shape)

        self.X_full = np.vstack([X_in1, X_left, X_right, X_top, X_bottom])

        percent = 50

        per_domain = 10

        N2 = int((percent/100) * len(X_in1))

        print('N2', N2)

        idx = np.random.choice(X_in1.shape[0], N2, replace=False)
        X_domain = X_in1[idx, :]

        N3 = int((per_domain / 100) * len(X_domain))

        print('N3', N3)

        idx1 = np.random.choice(X_domain.shape[0], N3, replace=False)

        X_domain_data = X_domain[idx1,:]

        plt.scatter(self.X_left[:,0], self.X_left[:,1], color='blue')
        plt.scatter(self.X_right[:, 0], self.X_right[:, 1], color='blue')
        plt.scatter(self.X_top[:, 0], self.X_top[:, 1], color='blue')
        plt.scatter(self.X_bottom[:, 0], self.X_bottom[:, 1], color='blue')
        plt.scatter(X_domain[:,0], X_domain[:,1], color='red')
        plt.scatter(X_domain_data[:,0], X_domain_data[:,1], color='orange')
        plt.show()

        V_p_train_domain, X_train_domain, V_p_test_domain, X_test_domain = self.train_test(X_domain_data, "domain")
        V_p_train_left, X_train_left, V_p_test_left, X_test_left = self.train_test(X_left, "left")
        V_p_train_right, X_train_right, V_p_test_right, X_test_right = self.train_test(X_right, "right")
        V_p_train_top, X_train_top, V_p_test_top, X_test_top = self.train_test(X_top, "top")
        V_p_train_bottom, X_train_bottom, V_p_test_bottom, X_test_bottom = self.train_test(X_bottom, "bottom")
        #V_p_train_full, X_train_full, V_p_test_full, X_test_full = self.train_test(X_in1, "full")


        domain_train = np.hstack([X_train_domain,V_p_train_domain])
        domain_test = np.hstack([X_test_domain, V_p_test_domain])

        #print(domain_train)

        #left_zeros = np.zeros((X_train_left.shape[0],1))

        left_train = np.hstack([X_train_left, V_p_train_left])
        left_test = np.hstack([X_test_left, V_p_test_left])

        #right_zeros = np.zeros((X_train_right.shape[0],1))

        right_train = np.hstack([X_train_right, V_p_train_right])
        right_test = np.hstack([X_test_right, V_p_test_right])

        #top_zeros = np.zeros((X_train_top.shape[0],1))

        top_train = np.hstack([X_train_top, V_p_train_top])
        top_test = np.hstack([X_test_top, V_p_test_top])

        #bottom_zeros = np.zeros((X_train_bottom.shape[0],1))

        bottom_train = np.hstack([X_train_bottom, V_p_train_bottom])
        bottom_test = np.hstack([X_test_bottom, V_p_test_bottom])

        #full_train = np.hstack([X_train_full, V_p_train_full])
        #full_test = np.hstack([X_test_full, V_p_test_full])

        u0 = domain_train[:,3]
        v0 = domain_train[:,4]

        X_physical = np.vstack([X_domain,X_train_left, X_train_right, X_train_top, X_train_bottom])
        U_grid = np.sqrt(np.square(u0) + np.square(v0))

        #plt.quiver(domain_train[:, 0], domain_train[:, 1], u0, v0, U_grid, scale=30)
        #plt.colorbar()
        print("data", X_physical.shape)
        #plt.show()

        self.V_p_star = np.vstack([self.u_star, self.v_star, self.p_star])

        u_full,v_full = self.velocity(self.X_full)
        p_full = self.pressure(self.X_full)

        #u_full, v_full, p_full = self.normalize(u_full, v_full, p_full)
        V_p_full = np.vstack([u_full, v_full, p_full])

        h5 = h5py.File('../data/data_Taylor_Green_Vortex_reduced'+str(per_domain)+'_'+str(t)+'.h5', 'w')
        g1 = h5.create_group('domain')
        g1.create_dataset('data1', data=domain_train)
        g1.create_dataset('data2', data=domain_test)

        g2 = h5.create_group('left')
        g2.create_dataset('data1', data=left_train)
        g2.create_dataset('data2', data=left_test)

        g3 = h5.create_group('right')
        g3.create_dataset('data1', data=right_train)
        g3.create_dataset('data2', data=right_test)

        g4 = h5.create_group('top')
        g4.create_dataset('data1', data=top_train)
        g4.create_dataset('data2', data=top_test)

        g5 = h5.create_group('bottom')
        g5.create_dataset('data1', data=bottom_train)
        g5.create_dataset('data2', data=bottom_test)

        g8 = h5.create_group('full')
        g8.create_dataset('data1', data=self.X_full)
        g8.create_dataset('data2', data=V_p_full)
        g8.create_dataset('data3', data=self.p_max)
        g8.create_dataset('data4', data=self.p_min)
        g8.create_dataset('data5', data=X_physical)

        h5.close()

def main():
    rho = 1.0
    nu = 0.01
    n = 0.9

    #create_data_list_csv()
    preprocessing = Preprocessing_Taylor_Green(rho, nu, n)
    #preprocessing.X_gen(1)
    preprocessing.data_generation()
    X_initial = preprocessing.X_full
    print('X', X_initial)
    u_initial, v_initial =preprocessing.velocity(X_initial)
    p_initial = preprocessing.pressure(X_initial)
    plotting(X_initial, u_initial, v_initial, p_initial)

def create_data_list_csv():
    directory = os.fsencode("../data/S2S/")
    labels=[]
    for file in os.listdir(directory):
        filename = str(file)
        label = str(re.search(r'\d+', filename).group())

        print(label)
        labels.append(int(label))
    labels_np = np.array(labels)
    np.savetxt("../data/timeSteps.csv", labels_np, delimiter=",")
def plotting(X, u, v, p):

    X_l = X
    u0 = u
    v0 = v
    p0 = p
    X_l1 = X[::5]
    u1 = u[::5]
    v1 = v[::5]

    p_min = p.min()
    p_max = p.max()
    u_max = u.max()
    v_max = v.max()
    scale = (u_max + v_max)/(2* p_max)
    #p0 = (-1 + 2*((p -p_min) / (p_max - p_min)))
    #p_orig = p_max*p0

    print(u0.shape)

    U_grid = np.sqrt(np.square(u1) + np.square(v1))

    """x_grid = np.reshape(X_l[:,0], (82,50))
    y_grid = np.reshape(X_l[:,1], (82,50))

    u_grid = np.reshape(u0, (82,50))
    v_grid = np.reshape(v0, (82,50))
    p_grid = np.reshape(p0, (82,50))

    u_grid_max = u_grid.max()
    u_grid_min = u_grid.min()

    v_grid_max = v_grid.max()
    v_grid_min = v_grid.min()

    p_grid_max = p_grid.max()
    p_grid_min = p_grid.min()

    print(p_grid_max)"""
    fig,ax = plt.subplots(1,1)
    a = ax.tricontourf(X_l[:,0],X_l[:,1],p0, levels=8, cmap='winter')
    b = ax.quiver(X_l1[:, 0], X_l1[:, 1], u1, v1, U_grid, scale=30, cmap='magma')
    cbar1 = plt.colorbar(a)
    cbar2 = plt.colorbar(b)
    """cbar1.set_ticklabels([])
    cbar2.set_ticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])"""
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    cbar1.set_label(r'$p$', rotation=0)
    cbar2.set_label(r'$U_{mag}$', rotation=0)
    plt.savefig('TGV.pdf', dpi=400)
    #tkz.save("TGV0.tex")
    plt.show()

    fig, ax = plt.subplots(1, 3)
    c1 = ax[0].tricontourf(X_l[:,0],X_l[:,1],u0, levels=7)
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('u_test', y=-0.1)


    c2 = ax[1].tricontourf(X_l[:,0],X_l[:,1],v0, levels=7)
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title('v_test', y=-0.1)


    c3 = ax[2].tricontourf(X_l[:,0],X_l[:,1],p0, levels=7)
    fig.colorbar(c3, ax=ax[2])
    ax[2].set_title('p_test', y=-0.1)

    '''c4 = ax[3].tricontourf(X_l[:, 0], X_l[:, 1], p_orig, levels=7)
    fig.colorbar(c4, ax=ax[3])
    ax[2].set_title('p_test_orig', y=-0.1)'''


    plt.show()

if __name__ == "__main__":
    main()
    sys.exit()




