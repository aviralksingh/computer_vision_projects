import torch
import torchvision.datasets as datasets # Standard datasets
from tqdm import tqdm
from torch import nn, optim
from model import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


INPUT_DIM= 784
H_DIM= 200
Z_DIM= 20

def inference (model):
    with torch.no_grad():
        z = torch.randn(64, 2)
        sample = model.decoder(z)
    save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')

trained_model = model= VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
trained_model.load_state_dict(torch.load('model/mnist_11.pth'))
for idx in range(10):
    inference(model)
