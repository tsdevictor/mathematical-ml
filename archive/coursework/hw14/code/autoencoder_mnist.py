import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from matplotlib import pyplot as plt
import seaborn as sns
import time

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, k):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, k),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(k, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 784),
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def loss(x,xout):
    return torch.mean((x-xout)**2)

def show_image(x, save_path):
    x = x.reshape(28,28)
    plt.imshow(x)
    plt.savefig(save_path)
    plt.show()

def pca(X, k, return_encoding=False):
    Xm = X - np.mean(X, axis=0)
    l,Q = eigh(Xm.T @ Xm)
    
    Q = Q[:,::-1]

    Q = Q[:,:k]
    X_encode = X @ Q
    if return_encoding:
        return X_encode
    X_decode = X_encode @ Q.T

    return X_decode


# parameters
batch_size = 50
learning_rate = 1e-4
num_epochs = 10
k = 2 # latent space dimension

# Data Preparation
d = pd.read_csv('mnist.csv', header=None)
y = d.iloc[:,0].to_numpy()
X = d.iloc[:,1:].to_numpy()
X = X/255

X_test = X[:5000,:]
X = X[500:,:]
y_test = y[:5000]
y = y[5000:]

X = torch.from_numpy(X).float()
X_test = torch.from_numpy(X_test).float()

# set up batches beforehand
all_samples = np.arange(X.shape[0]).reshape(batch_size, X.shape[0]//batch_size)
batches = [all_samples[:,i] for i in range(all_samples.shape[1])]

# Model, Loss, and Optimizer
model = Autoencoder(k)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
start_time = time.time()

# Training Loop
for epoch in range(num_epochs):
    for i, idx in enumerate(batches):
        x = X[idx,:]

        xout = model(x)
        l = loss(x,xout)
        losses.append(l.item())

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if i % 10 == 0: print([epoch, i, l.item()])

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('autoencoder/autoencoder_loss.png')
plt.show()

# look at the encoding
X_test = torch.from_numpy(X_test).float()
X_ae_encode = model.encoder(X_test).detach().numpy()
X_pca_encode = pca(X_test.detach().numpy(), k, return_encoding=True)

ys = [str(int(y)) for y in y_test]
sns.scatterplot(x=X_ae_encode[:,0], y=X_ae_encode[:,1], hue=ys)
plt.savefig('autoencoder/ae_encode.png')
plt.show()

sns.scatterplot(x=X_pca_encode[:,0], y=X_pca_encode[:,1], hue=ys)
plt.savefig('autoencoder/pca_encode.png')
plt.show()

# look at the fits
X_ae = model(X_test).detach().numpy()
X_pca = pca(X_test.detach().numpy(), k)
for i in range(10):
    show_image(X_test[i,:], 'autoencoder/imgs/'+str(i)+'_test.png')
    show_image(X_ae[i], 'autoencoder/imgs/'+str(i)+'_ae.png')
    show_image(X_pca[i], 'autoencoder/imgs/'+str(i)+'_pca.png')

X_test = X_test.detach().numpy()
error_ae = np.mean((X_test - X_ae)**2)
error_pca = np.mean((X_test - X_pca)**2)
print(f"AE error: {error_ae:.4f}, PCA error: {error_pca:.4f}")

