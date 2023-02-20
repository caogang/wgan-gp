from torch import optim
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import tflib.inception_score
import tflib.plot
import tflib.cat64x64
import tflib.save_images
import tflib as lib
import time
import os
import sys
sys.path.append(os.getcwd())


DATA_DIR = "cat64x64"
if os.path.exists(DATA_DIR):
    tflib.cat64x64.download_dataset(DATA_DIR)
if len(DATA_DIR) == 0:
    raise Exception("Please specify path to data directory in cat64x64.py!")

MODE = 'wgan-gp'
DIM = 128  # This overfits substantially; you're probably better off with 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 200000  # How many generator iterations to train for
OUTPUT_DIM = 3*64*64  # Number of pixels in Cat64x64 (3*64*64)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )

        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class DiscConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_activation=True):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
        )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.conv_block(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = DiscConvBlock(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = DiscConvBlock(64, 128, kernel_size=3, stride=2)
        self.conv3 = DiscConvBlock(128, 256, kernel_size=3, stride=2)
        self.conv4 = DiscConvBlock(256, 512, kernel_size=3, stride=2)
        self.conv5 = DiscConvBlock(512, 1, kernel_size=3, stride=1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)  
        out = out.view(out.shape[0], -1)
        return out


netG = Generator()
netD = Discriminator()
print(netG)
print(netD)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement(
    ) / BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1-alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    grad_output = torch.ones(disc_interpolates.size()).cuda(
        gpu) if use_cuda else torch.ones(disc_interpolates.size())

    gradients = autograd.grad(output=disc_interpolates, inputs=interpolates,
                              grad_outputs=grad_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_image(frame, netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)

    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 64, 64)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples, "./tmp/cat64x64/samples_{}.jpg".format(frame))

# For Calculating inception score


def get_inception_score(G):
    all_samples = []
    for _ in range(10):
        samples_100 = torch.randn(100, 128)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(
        np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype("int32")
    all_samples = all_samples.reshape((-1, 3, 64, 64)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))


# Dataset iterator
dataset = lib.cat64x64.CatDataset(DATA_DIR)  # Add custom transform if you want
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
test_size = dataset_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_gen = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dev_gen = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


for iteration in range(ITERS):
    start_time = time.time()

    D_cost = 0
    G_cost = 0
    Wasserstein_D = 0
    for i, x in enumerate(train_gen):
        if (i+1) % CRITIC_ITERS != 0:
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True
            _data = x
            netD.zero_grad()

            # train with real
            real_data = _data.reshape(
                BATCH_SIZE, 3, 64, 64).transpose(0, 2, 3, 1)

            if use_cuda:
                real_data = _data.cuda(gpu)
            real_data_v = autograd.Variable(real_data)

            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(BATCH_SIZE, 128)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(
                noise, volatile=True)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(
                netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = 0
            Wasserstein_D = 0
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        else:
            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = False
            netG.zero_grad()

            noise = torch.randn(BATCH_SIZE, 128)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise)
            fake = netG(noisev)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()

            # Write logs and save samples
            lib.plot.plot("./tmp/cat64x64/train disc cost",
                          D_cost.cpu().data.numpy())
            lib.plot.plot("./tmp/cat64x64/time", time.time() - start_time)
            lib.plot.plot("./tmp/cat64x64/train gen cost",
                          G_cost.cpu().data.numpy())
            lib.plot.plot("./tmp/cat64x64/wasserstein distance",
                          Wasserstein_D.cpu().data.numpy())

        # Calculate inception score every 1K iters
        if (iteration+1) % 1000 == 0:
            inception_score = get_inception_score(netG)
            lib.plot.plot('./tmp/cat64x64/inception score', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        if (iteration + 1) % 100 == 0:
            dev_disc_costs = []
            for images in dev_gen():
                images = images.reshape(
                    BATCH_SIZE, 3, 64, 64).transpose(0, 2, 3, 1)

                if use_cuda:
                    imgs = imgs.cuda(gpu)
                imgs_v = autograd.Variable(imgs, volatile=True)

                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot("./tmp/cat64x64/dev disc cost",
                          np.mean(dev_disc_costs))

            generate_image(iteration, netG)

        # Save Logs every 100 iters
        if (iteration < 5) or (iteration == 99):
            lib.plot.flush()
        lib.plot.tick()
