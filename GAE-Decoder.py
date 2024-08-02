# Standard library imports
import math
import pickle
import random
import time

# Third-party library imports
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.metrics import structural_similarity as ssim
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GAE
from tqdm import tqdm

def x_to_y(X): # averaging in 2*2 windows (4 pixels)
    dim = X.shape[0]
    dim = 20
    Y = np.zeros((int(dim/2),int(dim/2)))
    for i in range(int(dim/2)):
        for j in range(int(dim/2)):
            Y[i,j] = X[2*i,2*j] + X[2*i+1,2*j] + X[2*i,2*j+1] + X[2*i+1,2*j+1]

            Y_noise = np.random.multivariate_normal(np.zeros(100),0.0000 * np.eye(100))
            Y_noise.shape = (10,10)
            Y = Y + Y_noise
    return Y


class shallow(object):

    time = 0

    plt = []
    fig = []


    def __init__(self, x=[],y=[],h_ini = 1.,u=[],v = [],dx=0.01,dt=0.0001, N=64,L=1., px=16, py=16, R=64, Hp=0.1, g=1., b=0.): # How define no default argument before?


        # add a perturbation in pressure surface


        self.px, self.py = px, py
        self.R = R
        self.Hp = Hp



        # Physical parameters

        self.g = g
        self.b = b
        self.L=L
        self.N=N

        self.dx=dx
        self.dt=dt

        self.x,self.y = np.mgrid[:self.N,:self.N]

        self.u=np.zeros((self.N,self.N))
        self.v=np.zeros((self.N,self.N))

        self.h_ini=h_ini

        self.h=self.h_ini * np.ones((self.N,self.N))

        rr = (self.x-px)**2 + (self.y-py)**2
        self.h[rr<R] = self.h_ini + Hp #set initial conditions

        self.lims = [(self.h_ini-self.Hp,self.h_ini+self.Hp),(-0.02,0.02),(-0.02,0.02)]



    def dxy(self, A, axis=0):
        """
        Compute derivative of array A using balanced finite differences
        Axis specifies direction of spatial derivative (d/dx or d/dy)
        dA[i]/dx =  (A[i+1] - A[i-1] )  / 2dx
        """
        return (np.roll(A, -1, axis) - np.roll(A, 1, axis)) / (self.dx*2.) # roll: shift the array axis=0 shift the horizontal axis

    def d_dx(self, A):
        return self.dxy(A,1)

    def d_dy(self, A):
        return self.dxy(A,0)


    def d_dt(self, h, u, v):
        """
        http://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
        """
        for x in [h, u, v]: # type check
           assert isinstance(x, np.ndarray) and not isinstance(x, np.matrix)

        g,b,dx = self.g, self.b, self.dx

        du_dt = -g*self.d_dx(h) - b*u
        dv_dt = -g*self.d_dy(h) - b*v

        H = 0 #h.mean() - our definition of h includes this term
        dh_dt = -self.d_dx(u * (H+h)) - self.d_dy(v * (H+h))

        return dh_dt, du_dt, dv_dt


    def evolve(self):
        """
        Evolve state (h, u, v) forward in time using simple Euler method
        x_{N+1} = x_{N} +   dx/dt * d_t
        """

        dh_dt, du_dt, dv_dt = self.d_dt(self.h, self.u, self.v)
        dt = self.dt

        self.h += dh_dt * dt
        self.u += du_dt * dt
        self.v += dv_dt * dt
        self.time += dt

        return self.h, self.u, self.v

def simu(iteration_times, Hp, R, n_steps, blank_steps, px, py):
    SW = shallow(N=64, px=px, py=py, R=R, Hp=Hp, b=0.2)
    num = (iteration_times - blank_steps) // n_steps
    true_state_vect = np.zeros((num, SW.N, SW.N, 3))
    index = 0
    for i in range(iteration_times):
        SW.evolve()
        if i % n_steps == 0 and i >= blank_steps:
            true_state_vect[index] = np.dstack((SW.u, SW.v, SW.h))
            index += 1
    return true_state_vect

random.seed(112)
second_element_range = (0.2, 0.8)
third_element_range = (4, 12)
num_random_tuples = 40
steps = 3500
blank_steps = 500
px = 32
py = 32
sim_params = [(steps, round(random.uniform(*second_element_range), 2), random.randint(*third_element_range), 10, blank_steps, px, py) for _ in range(num_random_tuples)]

def simulate_and_stack(sim_params):
    true_state_vect = []
    for nsteps, hp, r, n, blank, px, py in sim_params:
        state = simu(nsteps, hp, r**2, n, blank, px, py)
        true_state_vect.append(state)
    true_state_vect = np.vstack(true_state_vect)
    return true_state_vect

#true_state_vect = simulate_and_stack(sim_params)
#plt.imshow(true_state_vect[100,:,:,0])
#plt.savefig('x.png')
import torch
import numpy as np
import random
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

def generate_graph_data(true_state_vect, radius=13):
    graphs = []
    for state in true_state_vect:
        # Define the indices for sensor placement and apply random movement
        base_indices = [(i, j) for i in range(0, 63, 7) for j in range(0, 63, 7)]
        moves = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        selected_indices = []

        for (i, j) in base_indices:
            move = random.choice(moves)
            new_i, new_j = i + move[0], j + move[1]
            selected_indices.append((max(0, min(new_i, 63)), max(0, min(new_j, 63))))

        # Extract features for selected sensor locations
        features = np.array([state[i, j, :] for (i, j) in selected_indices])

        # Create the graph with nodes and edges
        G = nx.Graph()
        for idx, ((i, j), feature) in enumerate(zip(selected_indices, features)):
            G.add_node(idx, pos=(i, j), color=feature)

        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                if np.linalg.norm(np.array(selected_indices[i]) - np.array(selected_indices[j])) <= radius:
                    G.add_edge(i, j)

        node_features = torch.tensor(features, dtype=torch.float)
        edge_list = list(G.edges())
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        batch = torch.zeros(len(selected_indices), dtype=torch.long)
        
        # Assuming 'image' is a part of the state or can be derived from it
        image = torch.tensor(state, dtype=torch.float)  # Adjust this line as necessary
        
        graphs.append(Data(x=node_features, edge_index=edge_index, batch=batch, image=image))
    
    return graphs
#graph_data_list = generate_graph_data(true_state_vect)


#torch.save(graph_data_list, '/data/nas/hanyang/test_sw/graph_data.pt')
graph_data_list = torch.load('/data/nas/hanyang/test_sw/graph_data.pt')
import torch
from torch import nn

import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

def custom_activation_graph(x):
    tanh_activation = torch.tanh(x[:, :2])
    scaled_feature_3 = F.relu(x[:, 2:])
    return torch.cat([tanh_activation, scaled_feature_3], dim=-1)
def custom_activation_image(x):
    tanh_activation = torch.tanh(x[:,:2,:,:])
    scaled_feature_3 = F.relu(x[:,2:, :,:])
    return torch.cat([tanh_activation, scaled_feature_3], dim=1)

'''class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.encoder_conv1 = GATConv(in_features, 16)
        self.encoder_conv2 = GATConv(16, 32)
        self.decoder_conv1 = GATConv(32, 16)
        self.decoder_conv2 = GATConv(16, in_features)

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.encoder_conv1(x, edge_index))
        x2 = F.relu(self.encoder_conv2(x1, edge_index))
        x_global = global_mean_pool(x2, batch)
        x_reconstructed = F.relu(self.decoder_conv1(x2, edge_index))
        x_reconstructed = self.decoder_conv2(x_reconstructed, edge_index)
        x_reconstructed = custom_activation_graph(x_reconstructed)  # Apply custom activation
        return x_reconstructed, x_global.view(-1, 32)
'''

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool,global_add_pool


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_features=3, latent_space=32):
        super().__init__()
        self.in_features = in_features
        self.latent_space = latent_space
        
        # Encoder layers
        self.encoder_conv1 = GATConv(self.in_features, 32)
        self.encoder_conv2 = GATConv(32, 64)
        self.encoder_conv3 = GATConv(64, self.latent_space)  # Using latent_space as the final encoder output size
        
        # Fully connected layer to expand the latent space back to node space
        self.fc_expand = Linear(self.latent_space, 81 * self.in_features)  # Expand from latent space
        
        # Decoder layers
        self.decoder_conv1 = GATConv(self.in_features, 64)  # Reverse order
        self.decoder_conv2 = GATConv(64, 32)  # Symmetric to encoder
        self.decoder_conv3 = GATConv(32, self.in_features)  # Output should match the input feature size

    def forward(self, x, edge_index, batch):
        # Encoder pass
        x1 = F.relu(self.encoder_conv1(x, edge_index))
        x2 = F.relu(self.encoder_conv2(x1, edge_index))
        x3 = F.relu(self.encoder_conv3(x2, edge_index))  # Final encoding to latent space
        x_global = global_mean_pool(x3, batch)
        
        # Feature expansion
        x_expanded = self.fc_expand(x_global)
        x_expanded = x_expanded.view(-1, self.in_features)  # Reshape back to match input feature dimension
        
        # Decoder pass
        x_reconstructed = F.relu(self.decoder_conv1(x_expanded, edge_index))
        x_reconstructed = F.relu(self.decoder_conv2(x_reconstructed, edge_index))
        x_reconstructed = self.decoder_conv3(x_reconstructed, edge_index)
        x_reconstructed = custom_activation_graph(x_reconstructed)
        
        return x_reconstructed, x_global.view(-1, self.latent_space)
                  

# Example usage
model = GraphAutoencoder(in_features=3, latent_space=64)

# Example usage
num_nodes = 81

'''class CompleteModel(torch.nn.Module):
    def __init__(self, in_features):
        super(CompleteModel, self).__init__()
        self.gae = GraphAutoencoder(in_features)

        # Image decoder
        self.image_decoder_dense = nn.Linear(32, 16*16*32)
        self.image_decoder_reshape = nn.Sequential(
                nn.Conv2d(32, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(16, 3, kernel_size=3, padding=1),
                nn.ReLU()
                )

    def forward(self, graph_data):
        x_reconstructed, latent_space = self.gae(graph_data.x, graph_data.edge_index, graph_data.batch)

        # Ensure latent_space has the expected shape
        latent_space = latent_space.view(-1, 32)
        # Image decoding
        image_reconstructed_flat = F.relu(self.image_decoder_dense(latent_space))
        image_reconstructed = self.image_decoder_reshape(image_reconstructed_flat.view(-1, 32, 16, 16))

        image_reconstructed = custom_activation_image(image_reconstructed)
        image_reconstructed = image_reconstructed.permute(0, 2, 3, 1)

        return x_reconstructed, image_reconstructed'''

class CompleteModel(torch.nn.Module):
    def __init__(self, in_features):
        super(CompleteModel, self).__init__()
        self.gae = GraphAutoencoder(in_features,64)

        # Image decoder
        self.image_decoder_dense = nn.Linear(64, 16*16*32)
        self.image_decoder_reshape = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            #nn.ReLU()
        )

    def forward(self, graph_data):
        x_reconstructed, latent_space = self.gae(graph_data.x, graph_data.edge_index, graph_data.batch)

        # Ensure latent_space has the expected shape
        latent_space = latent_space.view(-1, 64)
        # Image decoding
        image_reconstructed_flat = self.image_decoder_dense(latent_space)
        image_reconstructed = self.image_decoder_reshape(image_reconstructed_flat.view(-1, 32, 16, 16))
        image_reconstructed = custom_activation_image(image_reconstructed)
        image_reconstructed = image_reconstructed.permute(0, 2, 3, 1)  # Reshaping to fit image format
        return x_reconstructed, image_reconstructed

import torch
import torch.nn as nn
import torch.nn.functional as F



# Example usage
model = CompleteModel(in_features=3)

graph_data = graph_data_list[0]
x_reconstructed, image_reconstructed = model(graph_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CompleteModel(in_features=3).to(device)

print(model)

from torch.optim import Adam
from torch.nn import MSELoss

optimizer = Adam(model.parameters(), lr=0.01)
mse_loss = MSELoss()


# Prepare DataLoader
from torch_geometric.data import DataLoader


# Prepare DataLoader
if not isinstance(graph_data_list, list):
    graph_data_list = list(graph_data_list)
train_size = int(0.75 * len(graph_data_list))
train_dataset = graph_data_list[:train_size]
test_dataset = graph_data_list[train_size:]

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

def train(model, data_loader):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_reconstructed, image_reconstructed = model(data)
        target_shape = image_reconstructed.shape
        data.image = data.image.view(target_shape)
        feature_loss = mse_loss(x_reconstructed, data.x)
        image_loss = mse_loss(image_reconstructed, data.image)  # Assuming data.image is the target image
        
        loss = feature_loss + image_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(data_loader)
    print(f"Train function returning: {average_loss}")  # Check what is being returned
    return average_loss

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio

def test(model, data_loader, device):
    model.eval()
    model.to(device)
    total_loss = 0
    total_psnr = 0
    
    # Initialize PSNR metric and move it to the correct device
    

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            x_reconstructed, image_reconstructed = model(data)
            feature_loss = F.mse_loss(x_reconstructed, data.x)
            
            target_shape = image_reconstructed.shape
            data.image = data.image.view(target_shape).to(device)
            x_reconstructed = x_reconstructed.to(device)
            image_loss = F.mse_loss(image_reconstructed, data.image)

            data_range = torch.max(image_reconstructed) - torch.min(image_reconstructed)
            psnr_metric = PeakSignalNoiseRatio(data_range).to(device)
            # Calculate PSNR for the reconstructed image using the metric
            image_psnr = psnr_metric(image_reconstructed, data.image)
            total_psnr += image_psnr.item()
            
            # Combine the feature loss and image loss
            loss = feature_loss + image_loss
            total_loss += image_loss.item()

    # Compute average loss and PSNR over all batches
    average_loss = total_loss / len(data_loader)
    average_psnr = total_psnr / len(data_loader)

    print(f"Test function returning: Average Loss = {average_loss}, Average PSNR = {average_psnr}")
    
    return average_loss, average_psnr

# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练和测试模型
epochs = 20
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    test_loss, test_psnr = test(model, test_loader,device)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f},Test PSNR = {test_psnr:.4f}")



import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Assume model is trained and test_loader is available

# Function to reconstruct and save an image
def reconstruct_and_save_image(model, test_loader, device, sample_index=99):
    # Set model to evaluation mode
    model.eval()

    # Get the 100th test sample
    for i, data in enumerate(test_loader):
        if i == sample_index:
            test_data = data
            break

    # Move data to the correct device
    test_data.x = test_data.x.to(device)
    test_data.edge_index = test_data.edge_index.to(device)
    test_data.batch = test_data.batch.to(device)

    # Run the model to get the reconstructed image
    with torch.no_grad():
        _, reconstructed_image = model(test_data)

    # Convert the reconstructed image to a PIL image
    reconstructed_image = reconstructed_image.squeeze(0)  # Remove batch dimension if necessary
    reconstructed_image = reconstructed_image.cpu().numpy()

    # Display and save the image
    plt.imshow(reconstructed_image[5,:,:,1])
    plt.axis('off')
    plt.savefig('/data/nas/hanyang/test_sw/fig/reconstructed_image.png', bbox_inches='tight', pad_inches=0)
    plt.show()

# Assuming device is defined, and the model and test_loader are properly set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Call the function to reconstruct and save the image
reconstruct_and_save_image(model, test_loader, device, sample_index=99)
