# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from model import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM= 784
H_DIM= 200
Z_DIM= 20

NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4 #Karpathy constant

bs = 100
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# Configuration
model= VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2).to(DEVICE)
optimizer= optim.Adam(model.parameters(),lr=LR_RATE)

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')# yn (logxn) | (1-yn) *log(1-xn)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    torch.save(model.state_dict(), f"model/mnist_{epoch}.pth")


if __name__=="__main__":
    for epoch in range(1, 51):
        train(epoch)

# 1. In practice VAE's are typically trained by estimating the log variance not the std,
# this is for numerical stability and improves convergence of the results so your loss would go from:
# `- torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))` ->
# `-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()`
# (where log_var is the output of your encoder, also your missing a factor 0.5 for the numerically stable ELBO)
# Also, the ELBO is the Expectation of the reconstruction loss (the mean in this case) and the negative sum of the KL divergence

# 2. The ELBO (the loss) is based on a variational lower bound its not just a 2 losses stuck
#  together as such arbitrarily weighting the reconstruction loss and the KL divergence will give you unstable results,
# that being said your intuition was on the right path.
# VAEs are getting long in the tooth now and there are heavily improve versions that focus specifically on "explainable"
# if you want to understand them I would look at the
# Beta-VAE paper (which weights the KL divergence) then look into
# Disentagled VAE (see: "Structured Disentangled Representations", "Disentangling by Factorising") these methodologies force each
# "factor" into a normal Gaussian distribution rather than mixing the latent variables. T
# he result would be for the MNIST with a z dim of 10 each factor representing theoretically
# a variation of each number so sampling from each factor will give you "explainable" generations.

# 3. Finally your reconstruction loss should be coupled with your epsilon (your variational prior),
# typically (with some huge simplifications) MSE => epsilon ~ Gaussian Distribution, BCE => epislon ~ Bernoulli distribution
