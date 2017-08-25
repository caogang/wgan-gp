# WGAN-GP
An pytorch implementation of Paper "Improved Training of Wasserstein GANs".

# Prerequisites

Python, NumPy, SciPy, Matplotlib
A recent NVIDIA GPU

**A latest master version of Pytorch**

# Progress

- [x] gan_toy.py : Toy datasets (8 Gaussians, 25 Gaussians, Swiss Roll).(**Finished** in 2017.5.8)

- [x] gan_language.py : Character-level language model (Discriminator is using **nn.Conv1d**. Generator is using **nn.Conv1d**. **Finished** in 2017.6.23. Finished in 2017.6.27.)


- [x] gan_mnist.py : MNIST (**Running Results while Finished** in 2017.6.26. Discriminator is using **nn.Conv1d**. Generator is using **nn.Conv1d**.)

- [ ] gan_64x64.py: 64x64 architectures(**Looking forward to your pull request**)

- [x] gan_cifar.py: CIFAR-10(**Great thanks to [robotcator](https://github.com/caogang/wgan-gp/pull/18)**)

# Results

- [Toy Dataset](results/toy/)

  Some Sample Result, you can refer to the [results/toy/](results/toy/) folder for **details**.

  - **8gaussians 154500 iteration**

  ![frame1612](imgs/8gaussians_frame1545.jpg)

  - **25gaussians 48500 iteration**

    ![frame485](imgs/25gaussians_frame485.jpg)

  - **swissroll 69400 iteration**

  ![frame694](imgs/swissroll_frame694.jpg)

- [Mnist Dataset](results/mnist/)

  Some Sample Result, you can refer to the [results/mnist/](results/mnist/) folder for **details**.

  ![mnist_samples_91899](imgs/mnist_samples_91899.png)

  ![mnist_samples_91899](imgs/mnist_samples_92299.png)

  ![mnist_samples_91899](imgs/mnist_samples_92499.png)

  ![mnist_samples_199999](imgs/mnist_samples_199999.png)

- Billion Word Language Generation (Using CNN, character-level)

  Some Sample Result after 8699 epochs which is detailed in [sample](imgs/lang_samples_8699.txt)

  I haven't run enough epochs due to that this is very time-comsuming.

  > He moved the mat all out clame t
  >
  > A fosts of shores forreuid he pe
  >
  > It whith Crouchy digcloued defor
  >
  > Pamreutol the rered in Car inson
  >
  > Nor op to the lecs ficomens o fe
  >
  > In is a " nored by of the ot can
  >
  > The onteon I dees this pirder , 
  >
  > It is Brobes aoracy of " medurn 
  >
  > Rame he reaariod to thim atreast
  >
  > The stinl who herth of the not t
  >
  > The witl is f ont UAy Y nalence 
  >
  > It a over , tose sho Leloch Cumm

- [Cifar10 Dataset](results/cifar10/)

  Some Sample Result, you can refer to the [results/cifar10/](results/cifar10/) folder for **details**.

  ![mnist_samples_91899](imgs/cifar10_samples_80099.jpg)

# Acknowledge

Based on the implementation [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training) and [martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)
