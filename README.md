# WGAN-GP [WIP]
An pytorch implementation of Paper "Improved Training of Wasserstein GANs".

# Prerequisites

Python, NumPy, Pytorch, SciPy, Matplotlib, Pytorch
A recent NVIDIA GPU

# Progress

- [x] gan_toy.py : Toy datasets (8 Gaussians, 25 Gaussians, Swiss Roll).(**Finished** in 2017.5.8)
- [ ] gan_language.py : Character-level language model (Paused due to that `ConvBackward is not differentiable`, Problem is under solving)
- [ ] gan_mnist.py : MNIST
- [ ] gan_64x64.py: 64x64 architectures
- [ ] gan_cifar.py: CIFAR-10

# Results

- [Toy Dataset](results/toy/)

  Some Sample Result, you can refer to the [results/toy/](results/toy/) folder for **details**.

  - **8gaussians 154500 iteration**

  ![frame1612](imgs/8gaussians_frame1545.jpg)

  - **25gaussians 48500 iteration**

    ![frame485](imgs/25gaussians_frame485.jpg)

  - **swissroll 69400 iteration**

  ![frame694](imgs/swissroll_frame694.jpg)

  ​

  ​

# Acknowledge

Based on the tensorflow implementation [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
