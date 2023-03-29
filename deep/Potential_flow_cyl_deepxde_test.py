import deepxde as dde
import numpy as np
import pickle

length = 16
R = 2

precision_train = 15
precision_test = 30

weight_inner = 10
weight_outer = 30
epochs = 5000
l5 = 0.001
num_dense_layers = 4
num_dense_nodes = 100
activation = "tanh"

if dde.backend.backend_name == "pytorch":
    import torch


def pde(X,U):

    u_x = dde.grad.jacobian(U, X, i=0, j=0)
    u_y = dde.grad.jacobian(U, X, i=0, j=1)
    v_x = dde.grad.jacobian(U, X, i=1, j=0)
    v_y = dde.grad.jacobian(U, X, i=1, j=1)

    laplace  = u_x + v_y
    vorticity = v_x - u_y

    return [laplace, vorticity]

def potential_fn(x, y):

    Q = (pow((R), 2)) * math.pi * U  # Constant

    x1 = torch.square(x) + torch.square(y)

    phi = (U * x) + (Q / math.pi) * (torch.divide(x, x1))  ## Ux + Q/pi(x/x^2+y^2)

    return phi

def velocity_cartesian_vjp(X):
    if torch.is_tensor(X) != True:
        X = torch.from_numpy(X).to(device)
            # print(X.size())

    X = torch.split(X, 1, dim=1)


    x = X[0]
    y = X[1]

        # print(x.size())

    g = (x, y)

    v = torch.ones_like(x, device=device)


    phi, phi_x_y = torch.autograd.functional.vjp(potential_fn, g, v, create_graph=False)

    V_x = phi_x_y[0]

    V_y = phi_x_y[1]


    #print('vx',V_x)

    # print(V_y.size())

    V = torch.concat((V_x, V_y), dim=1)

    return V
    
## boundary definition

def boundary_outer(X, on_boundary):

    return on_boundary and outer.on_boundary(X)

def boundary_inner(X, on_boundary):

    return on_boundary and inner.on_boundary(X)

## neumann/dirichlet

## define geometry 

outer = dde.geometry.Rectangle([-length / 2.0, -length / 2.0] , [length / 2.0, length / 2.0])
inner = dde.geometry.Disk([0, 0], R)

geom = outer - inner

bc_inner = dde.icbc.DirichletBC(geom, velocity_cartesian_vjp, boundary_inner) 
bc_outer = dde.icbd.DirichletBC(geom, velocity_cartesian_vjp, boundary_outer)

## data generation

data = dde.data.PDE(geom, pde, [bc_inner, bc_outer], num_domain=5000, num_boundary=1000, solution=velocity_cartesian_vjp, num_test=500)

## network

net = dde.nn.FNN([2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform")

model = dde.Model(data, net)

loss_weights = [1, weight_inner, weight_outer]

## model compile and training

model.compile("adam", lr=lr, metrics=["12 relative error"], loss_weights=loss_weights)

losshistory, train_state = model.train(iterations=epochs)

## save model
dde.saveplot(localhistory, train_state, issave=True, isplot=True)

## prediction and post process

Nx = 200
Ny = Nx

# Grid points
xmin, xmax, ymin, ymax = [-length / 2, length / 2, -length / 2, length / 2]
plot_grid = np.mgrid[xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j]
points = np.vstack(
    (plot_grid[0].ravel(), plot_grid[1].ravel(), np.zeros(plot_grid[0].size))
)

points_2d = points[:2, :]
u = model.predict(points[:2, :].T)
u = u.reshape((Nx, Ny))

ide = np.sqrt(points_2d[0, :] ** 2 + points_2d[1, :] ** 2) < R
ide = ide.reshape((Nx, Nx))

u_exact = velocity_cartesian_vjp(points.T)
u_exact = u_exact.reshape((Nx, Ny))
diff = u_exact - u
error = np.linalg.norm(diff) / np.linalg.norm(u_exact)
print("Relative error = ", error)

## save data/results

result = [u, u_exact, ide, error]
f = open('./result/result_dde_pot_flow_cyl.pkl', 'wb')
pickle.dump(result, f)
f.close()


