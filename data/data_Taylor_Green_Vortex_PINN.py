import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

import sys, os, time, random, shutil


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
        t0 = np.zeros((x.flatten().shape))
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


        self.X_initial = np.vstack([x.flatten(), y.flatten(), t0]).T
        self.X_bottom = np.vstack([x1.flatten(), X0, t1.flatten()]).T
        self.X_top = np.vstack([x1.flatten(), Xpi, t1.flatten()]).T
        self.X_left = np.vstack([X0, x1.flatten(), t1.flatten()]).T
        self.X_right = np.vstack([Xpi, x1.flatten(), t1.flatten()]).T
        self.X_center = np.vstack([center.flatten(), center.flatten(), t.flatten()]).T

        self.u_center, self.v_center = self.velocity(self.X_center)
        self.p_center = self.pressure(self.X_center)

        self.l1 = len(x.flatten())

        '''print(self.u_center)
        print(self.v_center)'''

        X = self.X_in[:self.l1]
        u = self.u_star[:self.l1]
        v = self.v_star[:self.l1]

        plt.quiver(self.X_center[:, 0], self.X_center[:, 1], self.u_center, self.v_center)
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

        X_in1 = self.X_in[self.l1:]

        N2 = int(0.3 * len(X_in1))

        idx = np.random.choice(X_in1.shape[0], N2, replace=False)
        X_domain = X_in1[idx, :]


        V_p_train_domain, X_train_domain, V_p_test_domain, X_test_domain = self.train_test(X_domain)
        V_p_train_left, X_train_left, V_p_test_left, X_test_left = self.train_test(self.X_left)
        V_p_train_right, X_train_right, V_p_test_right, X_test_right = self.train_test(self.X_right)
        V_p_train_top, X_train_top, V_p_test_top, X_test_top = self.train_test(self.X_top)
        V_p_train_bottom, X_train_bottom, V_p_test_bottom, X_test_bottom = self.train_test(self.X_bottom)
        V_p_train_initial, X_train_initial, V_p_test_initial, X_test_initial = self.train_test(self.X_initial)


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

        '''plt.quiver(X_train_initial[:, 0], X_train_initial[:, 1], V_p_train_initial[:,0], V_p_train_initial[:,1])
        plt.show()'''

        V_p_star = np.vstack([self.u_star, self.v_star, self.p_star])
        V_p_center = np.vstack([self.u_center, self.v_center, self.p_center])

        center_data = np.hstack([self.X_center, V_p_center.T])

        print(center_data.shape)

        #print(X_train_domain, V_p_train_domain)
        '''print(V_p_test_domain.shape)
        print(X_train_domain)
        print(X_test_domain.shape)'''

        h5 = h5py.File('data_Taylor_Green_Vortex.h5', 'w')
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
        g8.create_dataset('data2', data=V_p_star.T)

        h5.close()

def main():
    rho = 1.2
    nu = 1.516e-5
    n = 0.9

    preprocessing = Preprocessing_Taylor_Green(rho, nu, n)
    preprocessing.data_generation()

if __name__ == "__main__":
    main()
    sys.exit()




