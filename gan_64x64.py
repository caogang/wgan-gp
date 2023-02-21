from torch import optim
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import torchvision
import torch
import numpy as np
import tqdm
import pandas as pd
import tflib.cat64x64
import time
import os
import sys
import gc

sys.path.append(os.getcwd())


DATA_DIR = "cat64x64/cats"
if not os.path.exists(DATA_DIR):
    tflib.cat64x64.download_dataset(DATA_DIR)
if len(DATA_DIR) == 0:
    raise Exception("Please specify path to data directory in cat64x64.py!")

MODE = 'wgan-gp'
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 128  # Batch size
ITERS = 200000  # How many generator iterations to train for

nc = 3  # Number of image channel
nz = 128  # This overfits substantially; you're probably better off with 64
ngf = 64
ndf = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


netG = Generator()
netD = Discriminator()

# Initialize network weights
netG.apply(initialize_weights)
netD.apply(initialize_weights)

print(netG)
print(netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)


optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4, momentum=0.5)
optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4, momentum=0.5)


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand((BATCH_SIZE, int(real_data.nelement(
    ) / BATCH_SIZE))).contiguous().view(BATCH_SIZE, 3, 64, 64)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1-alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    grad_output = torch.ones(disc_interpolates.size()).cuda(
        gpu) if use_cuda else torch.ones(disc_interpolates.size())

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    del interpolates
    del alpha
    del disc_interpolates

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128, 1, 1)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)

    with torch.no_grad():
        noisev = autograd.Variable(fixed_noise_128)
        samples = netG(noisev)
        samples = samples.view(-1, 3, 64, 64)
        samples = samples.mul(0.5).add(0.5)
        # samples = samples.cpu().data.numpy()

        if not os.path.exists("./tmp/cat64x64/"):
            os.makedirs("./tmp/cat64x64/")
        save_image(samples, "./tmp/cat64x64/samples_{}.jpg".format(frame),
                   nrow=16, padding=2)


# Dataset iterator
# Add custom transform if you want
dataset = tflib.cat64x64.CatDataset(DATA_DIR)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
test_size = dataset_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_gen = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dev_gen = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

D_Losses = []
G_Losses = []

for iteration in range(ITERS):
    start_time = time.time()
    print(f"Epoch: [{iteration}/{ITERS}]")

    pbar = tqdm.tqdm(train_gen)
    _D_cost = 0

    for i, real_data in enumerate(pbar):
        pbar.set_description(f"Processing: {i}")

        if (i+1) % CRITIC_ITERS != 0:
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True

            netD.zero_grad()

            if use_cuda:
                real_data = real_data.cuda(gpu)
            real_data_v = autograd.Variable(real_data)
            D_real = torch.mean(netD(real_data_v))

            # train with fake
            noise = torch.randn(BATCH_SIZE, 128, 1, 1)
            if use_cuda:
                noise = noise.cuda(gpu)

            with torch.no_grad():
                noisev = autograd.Variable(noise)  # totally freeze netG
                fake = autograd.Variable(netG(noisev).data)

            D_fake = netD(fake)
            D_fake = torch.mean(D_fake)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(
                netD, real_data_v.data, fake.data)

            D_cost = D_fake - D_real + gradient_penalty
            D_cost.backward(retain_graph=True)
            optimizerD.step()
            pbar.set_postfix(D_cost=D_cost.item())

            _D_cost = D_cost.item()

            del real_data_v
            del fake
            del noisev
            del gradient_penalty
            del D_cost
            torch.cuda.empty_cache()
            gc.collect()

        else:
            ############################
            # (2) Update G network
            ###########################
            # for p in netD.parameters():  # reset requires_grad
            # p.requires_grad = False
            netD.eval()
            netG.zero_grad()

            noise = torch.randn(BATCH_SIZE, 128, 1, 1)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise)
            fake = netG(noisev)
            G_cost = -torch.mean(netD(fake))
            G_cost.backward()
            optimizerG.step()
            pbar.set_postfix(G_cost=G_cost.item())

            G_Losses.append(G_cost.item())
            D_Losses.append(_D_cost)

            del fake
            del noisev
            del noise
            del G_cost
            torch.cuda.empty_cache()
            gc.collect()

    # Generate samples every 100 iters
    if iteration < 5 or (iteration + 1) % 10 == 0:
        generate_image(iteration, netG)

    loss_df = pd.DataFrame({"G": G_Losses, "D": D_Losses})
    loss_df.to_csv("loss.csv")
