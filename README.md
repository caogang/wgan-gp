# WGAN-GP
An pytorch implementation of Paper "Improved Training of Wasserstein GANs".

# Prerequisites

Python, NumPy, Pytorch, SciPy, Matplotlib, Pytorch
A recent NVIDIA GPU

# Progress

- [x] gan_toy.py : Toy datasets (8 Gaussians, 25 Gaussians, Swiss Roll).(**Finished** in 2017.5.8)

- [x] gan_language.py : Character-level language model (**Running Results while Finished** in 2017.5.10 using **nn.Linear** instead of **nn.Conv1d**)

> Due to that `ConvBackward is not differentiable`, Problem is under solving. When the features are supported, i will change it back.

- [x] gan_mnist.py : MNIST (**Running Results while Finished** in 2017.5.11 using **nn.Linear** instead of **nn.Conv1d** in Discriminator)

- [ ] gan_64x64.py: 64x64 architectures(Due to time problem, this module will be delayed. **Looking forward to your pull request**)

- [ ] gan_cifar.py: CIFAR-10(Due to time problem, this module will be delayed. **Looking forward to your pull request**)

# Results

- [Toy Dataset](results/toy/)

  Some Sample Result, you can refer to the [results/toy/](results/toy/) folder for **details**.

  - **8gaussians 154500 iteration**

  ![frame1612](imgs/8gaussians_frame1545.jpg)

  - **25gaussians 48500 iteration**

    ![frame485](imgs/25gaussians_frame485.jpg)

  - **swissroll 69400 iteration**

  ![frame694](imgs/swissroll_frame694.jpg)

- [Mnist Dataset](result/mnist/)

  Some Sample Result, you can refer to the [results/mnist/](results/mnist/) folder for **details**.

  ![mnist_samples_91899](imgs/mnist_samples_91899.png)

  ![mnist_samples_91899](imgs/mnist_samples_92299.png)

  ![mnist_samples_91899](imgs/mnist_samples_92499.png)



# Acknowledge

Based on the implementation [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training) and [martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)
