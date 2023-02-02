import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from torch import nn
import argparse
import sys, os, time, random, shutil
import torch.distributed as dist
import pickle

torch.manual_seed(1234)

np.random.seed(1234)
class Preprocessing_Taylor_Green():
    def __init__(self, rho, nu, n):

        self.rho = rho
        self.nu = nu
        self.n = n

        x_values = np.arange(0, np.pi, 0.2).tolist()
        y_values = np.arange(0, np.pi, 0.2).tolist()
        t = np.arange(0, 1, 0.02)
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


        self.u_star, self.v_star = self.velocity(self.X_in)
        self.p_star = self.pressure(self.X_in)

        self.u_min = self.u_star.min()
        self.u_max = self.u_star.max()
        self.v_min = self.v_star.min()
        self.v_max = self.v_star.max()
        self.p_min = self.p_star.min()
        self.p_max = self.p_star.max()

        self.X_bottom = np.vstack([x1.flatten(), X0, t1.flatten()]).T
        self.X_top = np.vstack([x1.flatten(), Xpi, t1.flatten()]).T
        self.X_left = np.vstack([X0, x1.flatten(), t1.flatten()]).T
        self.X_right = np.vstack([Xpi, x1.flatten(), t1.flatten()]).T

        self.l1 = len(x.flatten())

        X = self.X_in[:self.l1]
        u = self.u_star[:self.l1]
        v = self.v_star[:self.l1]

        plt.quiver(X[:, 0], X[:, 1], u, v)
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
            u0 = np.sin(X[i][0])*np.cos(X[i][1])*F
            v0 = -np.cos(X[i][0])*np.sin(X[i][1])*F
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
            f = -2 * self.nu * X[i][2]
            F = np.exp(f)
            p0 = self.rho*(np.cos(2*X[i][0]) + np.sin(2*X[i][1]))*F
            p.append(p0)

        p_numpy = np.array(p)

        return p_numpy

    def normalize(self, u,v, p):
        u_norm = -1 + 2*((u - self.u_min) / (self.u_max - self.u_min))
        v_norm = -1 + 2*((v - self.v_min) / (self.v_max - self.v_min))
        p_norm = -1 + 2*((p -self.p_min) / (self.p_max - self.p_min))
        return u_norm, v_norm, p_norm

    def train_test(self,X):
        N = int(self.n * len(X))

        idx = np.random.choice(X.shape[0], N, replace=False)

        X_train = X[idx,:]

        X_test = np.delete(X, idx, axis=0)

        u_train, v_train = self.velocity(X_train)
        u_test, v_test = self.velocity(X_test)

        p_train = self.pressure(X_train)
        p_test = self.pressure(X_test)

        u_train_norm, v_train_norm, p_train_norm = self.normalize(u_train, v_train,p_train)

        u_test_norm, v_test_norm, p_test_norm = self.normalize(u_test, v_test, p_test)

        V_p_train = np.vstack([u_train_norm, v_train_norm, p_train_norm])
        V_p_test = np.vstack([u_test_norm, v_test_norm, p_test_norm])


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

        Domain_train = np.hstack([X_train_domain,V_p_train_domain])
        Domain_test = np.hstack([X_test_domain, V_p_test_domain])

        Left_train = np.hstack([X_train_left, V_p_train_left])
        Left_test = np.hstack([X_test_left, V_p_test_left])

        Right_train = np.hstack([X_train_right, V_p_train_right])
        Right_test = np.hstack([X_test_right, V_p_test_right])

        Top_train = np.hstack([X_train_top, V_p_train_top])
        Top_test = np.hstack([X_test_top, V_p_test_top])

        Bottom_train = np.hstack([X_train_bottom, V_p_train_bottom])
        Bottom_test = np.hstack([X_test_bottom, V_p_test_bottom])

        V_p_star = np.vstack([self.u_star, self.v_star, self.p_star])


        '''for i in range(len(X_train_domain)):
            print(X_train_domain[i], V_p_train_domain[i])'''

        #print(X_train_domain, V_p_train_domain)
        '''print(V_p_test_domain.shape)
        print(X_train_domain)
        print(X_test_domain.shape)'''

        h5 = h5py.File('data_Taylor_Green_Vortex_reduced.h5', 'w')
        g1 = h5.create_group('domain')
        g1.create_dataset('data1', data=Domain_train)
        g1.create_dataset('data2', data=Domain_test)

        g2 = h5.create_group('left')
        g2.create_dataset('data1', data=Left_train)
        g2.create_dataset('data2', data=Left_test)

        g3 = h5.create_group('right')
        g3.create_dataset('data1', data=Right_train)
        g3.create_dataset('data2', data=Right_test)

        g4 = h5.create_group('top')
        g4.create_dataset('data1', data=Top_train)
        g4.create_dataset('data2', data=Top_test)

        g5 = h5.create_group('bottom')
        g5.create_dataset('data1', data=Bottom_train)
        g5.create_dataset('data2', data=Bottom_test)

        g6 = h5.create_group('full')
        g6.create_dataset('data1', data=self.X_in)
        g6.create_dataset('data2', data=V_p_star.T)

        h5.close()

def main():
    rho = 1.2
    nu = 1.516e-5
    n = 0.8

    preprocessing = Preprocessing_Taylor_Green(rho, nu, n)
    preprocessing.data_generation()

if __name__ == "__main__":
    main()
    sys.exit()




