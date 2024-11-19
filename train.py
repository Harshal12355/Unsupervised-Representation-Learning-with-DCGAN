import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dcgan_model import Discriminator, Generator, initalize_weights

# Hyperparameters
lr = 0.0002
batch_size = 128
image_size = 64
img_channels = 1  # For grayscale images, use 1; for RGB, use 3.
noise_dim = 100
feature_gen = 64
feature_disc = 64
num_epochs = 5

# Loading MNIST Dataset
# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])  # Normalize to [-1, 1]
])

dataset = datasets.MNIST(root="data/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Loading CelebA Dataset
# # Data loading and preprocessing
# # Define a custom dataset
# class CelebADataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image

# # Define transformations
# image_size = 64
# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Initialize dataset and dataloader
# dataset = CelebADataset(root_dir='C:/Users/harsh/Desktop/ML Projects/Unsupervised Representation Learning with DCGAN/img_align_celeba', transform=transform)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator(noise_dim, img_channels, feature_gen).to(device)
disc = Discriminator(img_channels, feature_disc).to(device)
initalize_weights(gen)
initalize_weights(disc)

# Optimizers
optimizer_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss
criterion = nn.BCELoss()
