import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from sklearn.preprocessing import MinMaxScaler
import sys, os, time, random, shutil


torch.manual_seed(1234)

np.random.seed(1234)
class Preprocessing_Taylor_Green():
    def __init__(self, rho, nu, n):

        self.rho = rho
        self.nu = nu
        self.n = n

        x_values = np.arange(-np.pi, np.pi, 0.4).tolist()
        y_values = np.arange(-np.pi, np.pi, 0.4).tolist()
        self.t = np.arange(0, 100, 1)
        x_values.append(np.pi)
        y_values.append(np.pi)


        #y0,x0 = np.meshgrid(y_values, x_values)

        #X_in = np.hstack([x.flatten()[:, None], y.flatten()[:, None]])


        #y2, t2 = np.meshgrid(y0,t)

        x, y = np.meshgrid(x_values, y_values)
        self.X_in = np.hstack([x.flatten[:None], y.flatten[:None]])
        X0 = np.zeros(y.flatten().shape)
        Xpi = np.zeros(y.flatten().shape)
        Xpi[:] = np.pi
        #t0 = np.zeros((x.flatten().shape))
        #center = np.zeros(t.flatten().shape)
        #center[:] = np.pi/2

        #self.X_initial = np.vstack([x.flatten(), y.flatten(), t0]).T
        self.X_bottom = np.hstack([x.flatten(), X0])
        self.X_top = np.hstack([x.flatten(), Xpi])
        self.X_left = np.hstack([X0, x.flatten()])
        self.X_right = np.hstack([Xpi, x.flatten()])
        #self.X_center = np.vstack([center.flatten(), center.flatten(), t.flatten()]).T

        #self.u_center, self.v_center = self.velocity(self.X_center)
        #self.p_center = self.pressure(self.X_center)

        #self.l1 = len(x.flatten())

        '''print(self.u_center)
        print(self.v_center)'''


        '''plt.quiver(self.X_center[:, 0], self.X_center[:, 1], self.u_center, self.v_center)
        plt.show()'''


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
        scale = (self.u_max + self.v_max)/(2*self.p_max)
        print('Scale: ', 1/scale)
        p_norm = (-1 + 2*((p -self.p_min) / (self.p_max - self.p_min)))/scale
        return u_norm, v_norm, p_norm

    def denormalize(self, u_norm, v_norm, p_norm):
        u = (((u_norm + 1) * (self.u_max - self.u_min)) / 2 ) + self.u_min
        v = (((v_norm + 1) * (self.v_max - self.v_min)) / 2) + self.v_min
        scale = (self.u_max + self.v_max) / (2 * self.p_max)
        p = (((scale * p_norm + 1) * (self.p_max - self.p_min)) / 2) + self.p_min
        return u, v, p

    def train_test(self,X, flag):
        N = int(self.n * len(X))

        idx = np.random.choice(X.shape[0], N, replace=False)

        X_train = X[idx,:]

        X_test = np.delete(X, idx, axis=0)

        u_train, v_train = self.velocity(X_train)
        u_test, v_test = self.velocity(X_test)

        p_train = self.pressure(X_train)
        p_test = self.pressure(X_test)

        #u_train_norm, v_train_norm, p_train_norm = self.normalize(u_train, v_train,p_train)

        #u_test_norm, v_test_norm, p_test_norm = self.normalize(u_test, v_test, p_test)

        if flag == "domain":
            flag_mark = 1
        elif flag == "left" or flag == "right" or flag == "top" or flag == "bottom":
            flag_mark = 2
        elif flag == "initial":
            flag_mark = 0

        flag_train = torch.full((u_train.shape), flag_mark)
        flag_test = torch.full((u_test.shape), flag_mark)

        V_p_train = np.vstack([u_train, v_train, p_train, flag_train])
        V_p_test = np.vstack([u_test, v_test, p_test, flag_test])


        return V_p_train.T, X_train, V_p_test.T, X_test

    def data_generation(self):

        for t in self.t:

            t_in = np.zeros((self.X_in.shape[0],1))
            t_in[:] = t

            t_left = np.zeros((self.X_left.shape[0],1))
            t_left[:] = t
            X_in1 = np.vstack([self.X_in, ])

            N2 = int(0.3 * len(X_in1))

            idx = np.random.choice(X_in1.shape[0], N2, replace=False)
            X_domain = X_in1[idx, :]


            V_p_train_domain, X_train_domain, V_p_test_domain, X_test_domain = self.train_test(X_domain, "domain")
            V_p_train_left, X_train_left, V_p_test_left, X_test_left = self.train_test(self.X_left, "left")
            V_p_train_right, X_train_right, V_p_test_right, X_test_right = self.train_test(self.X_right, "right")
            V_p_train_top, X_train_top, V_p_test_top, X_test_top = self.train_test(self.X_top, "top")
            V_p_train_bottom, X_train_bottom, V_p_test_bottom, X_test_bottom = self.train_test(self.X_bottom, "bottom")
            V_p_train_initial, X_train_initial, V_p_test_initial, X_test_initial = self.train_test(self.X_initial, "initial")


            domain_train = np.hstack([X_train_domain,V_p_train_domain])
            domain_test = np.hstack([X_test_domain, V_p_test_domain])

            left_train = np.hstack([X_train_left, V_p_train_left])
            left_test = np.hstack([X_test_left, V_p_test_left])

            right_train = np.hstack([X_train_right, V_p_train_right])
            right_test = np.hstack([X_test_right, V_p_test_right])

            top_train = np.hstack([X_train_top, V_p_train_top])
            top_test = np.hstack([X_test_top, V_p_test_top])

            bottom_train = np.hstack([X_train_bottom, V_p_train_bottom])
            bottom_test = np.hstack([X_test_bottom, V_p_test_bottom])

            initial_train = np.hstack([X_train_initial, V_p_train_initial])
            initial_test = np.hstack([X_test_initial, V_p_test_initial])

            u0 = domain_train[:,3]
            v0 = domain_train[:,4]

            U_grid = np.sqrt(np.square(u0) + np.square(v0))

            plt.quiver(domain_train[:, 0], domain_train[:, 1], u0, v0, U_grid, scale=30)
            plt.colorbar()
            plt.show()

            self.V_p_star = np.vstack([self.u_star, self.v_star, self.p_star])
            V_p_center = np.vstack([self.u_center, self.v_center, self.p_center])

            center_data = np.hstack([self.X_center, V_p_center.T])

            #print(initial_train)

            #print(X_train_domain, V_p_train_domain)
            '''print(V_p_test_domain.shape)
            print(X_train_domain)
            print(X_test_domain.shape)'''

            h5 = h5py.File('data_Taylor_Green_Vortex_reduced_noNorm.h5', 'w')
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

            g6 = h5.create_group('initial')
            g6.create_dataset('data1', data=initial_train)
            g6.create_dataset('data2', data=initial_test)

            g7 = h5.create_group('center')
            g7.create_dataset('data1', data=center_data)

            g8 = h5.create_group('full')
            g8.create_dataset('data1', data=self.X_in)
            g8.create_dataset('data2', data=self.V_p_star.T)

            h5.close()

def main():
    rho = 1.2
    nu = 1.516e-5
    n = 0.9

    preprocessing = Preprocessing_Taylor_Green(rho, nu, n)
    preprocessing.data_generation()
    X_initial = preprocessing.X_initial
    u_initial, v_initial =preprocessing.velocity(X_initial)
    p_initial = preprocessing.pressure(X_initial)
    plotting(X_initial, u_initial, v_initial, p_initial)

def plotting(X, u, v, p):

    X_l = X
    u0 = u
    v0 = v
    p0 = p
    p_min = p.min()
    p_max = p.max()
    u_max = u.max()
    v_max = v.max()
    scale = (u_max + v_max)/(2* p_max)
    #p0 = (-1 + 2*((p -p_min) / (p_max - p_min)))/scale

    print(u0.shape)

    U_grid = np.sqrt(np.square(u0) + np.square(v0))

    x_grid = np.reshape(X_l[:,0], (9,9))
    y_grid = np.reshape(X_l[:,1], (9,9))

    u_grid = np.reshape(u0, (9,9))
    v_grid = np.reshape(v0, (9,9))
    p_grid = np.reshape(p0, (9,9))

    u_grid_max = u_grid.max()
    u_grid_min = u_grid.min()

    v_grid_max = v_grid.max()
    v_grid_min = v_grid.min()

    p_grid_max = p_grid.max()
    p_grid_min = p_grid.min()

    print(p_grid_max)

    plt.pcolormesh(x_grid, y_grid, p_grid, shading='gouraud', label='u_x_pred', vmin=p_grid_min,
                   vmax=p_grid_max, cmap=plt.get_cmap('rainbow'))
    plt.quiver(X_l[:, 0], X_l[:, 1], u0, v0, U_grid, scale=30)
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots(1, 3)
    c1 = ax[0].pcolormesh(x_grid, y_grid, u_grid, shading='gouraud', label='u_x_exact',
                             vmin=u_grid_min, vmax=u_grid_max, cmap=plt.get_cmap('rainbow'))
    fig.colorbar(c1, ax=ax[0])
    ax[0].set_title('u_test', y=-0.1)


    c2 = ax[1].pcolormesh(x_grid, y_grid, v_grid, shading='gouraud', label='v_x_exact',
                             vmin=v_grid_min, vmax=v_grid_max, cmap=plt.get_cmap('rainbow'))
    fig.colorbar(c2, ax=ax[1])
    ax[1].set_title('v_test', y=-0.1)


    c3 = ax[2].pcolormesh(x_grid, y_grid, p_grid, shading='gouraud', label='u_x_pred', vmin=p_grid_min,
                             vmax=p_grid_max, cmap=plt.get_cmap('rainbow'))
    fig.colorbar(c3, ax=ax[2])
    ax[2].set_title('p_test', y=-0.1)


    plt.show()

if __name__ == "__main__":
    main()
    sys.exit()




