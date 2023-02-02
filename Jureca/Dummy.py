import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)

            a = self.activation2(z)

        a = self.linears[-2](a)

        a = self.activation(a)

        a = self.linears[-1](a)


        print("\tIn Model: input size", X.size(),
              "output size", a.size())

        return a

def scaling(X):

    mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
    # preprocessing input
    x = (X - mean) / (std)  # feature scaling

    return x

class RandomData(Dataset):

    def __init__(self, list_id):
        self.len = 100
        self.data = torch.randn(self.len, 2)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layers = np.array([2, 60, 60, 2])
    model = Taylor_green_vortex_PINN(layers)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model.to(device)    

    input_size= 2

    list_id = np.arange(0,100)

    rand_loader  = DataLoader(dataset=RandomData(x for x in list_id), batch_size=30, shuffle=True)

    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(),
             "output_size", output.size())


main()        
