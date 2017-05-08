# WGAN-GP [WIP]
An pytorch implementation of Paper "Improved Training of Wasserstein GANs".

# Prerequisites

Python, NumPy, Pytorch, SciPy, Matplotlib
A recent NVIDIA GPU

# Process

- [x] gan_toy.py(**Finished** in 2017.5.8)
- [ ] gan_language.py(Working Progress)

# Results

## Toy Dataset

### **8gaussians** Dataset Training after **99799 generator iterators** 

1. Learning distribution

![8gaussians_99799](results/8gaussians_99799.jpg)

2. Discriminator Objective Function Curve

   ![8gaussians_disc_cost](results/8gaussians_disc_cost.jpg)

3. Generator Objective Function Curve

   ![8gaussians_gen_cost](results/8gaussians_gen_cost.jpg)