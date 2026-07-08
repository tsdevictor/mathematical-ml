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
class Encoder(nn.Module):
    def __init__(self, k):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, k),
        )

        self.predicter = nn.Sequential(
            nn.Linear(k, 10),
            nn.Softmax()
        )
    
    def predict(self, x):
        x = self.encoder(x)
        p = self.predicter(x)
        return p

def loss(p,hot_y):
    l = -1*torch.sum(hot_y * torch.log(p + 1E-6))
    return l

def pca(X, k):
    Xm = X - np.mean(X, axis=0)
    l,Q = eigh(Xm.T @ Xm)

    Q = Q[:,::-1]

    Q = Q[:,:k]
    X_encode = X @ Q
    return X_encode

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
X = X[5000:,:]
y_test = y[:5000]
y = y[5000:]

X = torch.from_numpy(X).float()
X_test = torch.from_numpy(X_test).float()
y = torch.from_numpy(y)
y_test = torch.from_numpy(y_test)
hot_y = nn.functional.one_hot(y, num_classes=10)
hot_y_test = nn.functional.one_hot(y_test, num_classes=10)

# set up batches beforehand
all_samples = np.arange(X.shape[0]).reshape(batch_size, X.shape[0]//batch_size)
batches = [all_samples[:,i] for i in range(all_samples.shape[1])]

# Model, Loss, and Optimizer
model = Encoder(k)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
start_time = time.time()

# Training Loop
for epoch in range(num_epochs):
    for i, idx in enumerate(batches):
        x = X[idx,:]
        y_b = hot_y[idx,:]

        xout = model.predict(x)
        l = loss(xout,y_b)
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
plt.savefig('supervised_encoder/supervised_loss.png')
plt.show()

X_ae = model.encoder(X_test).detach().numpy()
X_pca = pca(X_test.detach().numpy(), k)

ys = [str(int(y)) for y in y_test.numpy()]
sns.scatterwplot(x=X_ae[:,0], y=X_ae[:,1],hue=ys)
plt.savefig('supervised_encoder/ae_encode.png')
plt.show()

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1],hue=ys)
plt.savefig('supervised_encoder/pca_encode.png')
plt.show()
