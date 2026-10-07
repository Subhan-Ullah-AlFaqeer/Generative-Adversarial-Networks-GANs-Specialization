# 👁️ Week 2: Deep Convolutional GANs (DCGAN)

Welcome to Week 2 of **Build Basic Generative Adversarial Networks (GANs)** (Course 1 of the **DeepLearning.AI GANs Specialization**)! This module transitions from flat fully-connected networks to spatial deep convolutional architectures designed specifically for complex image synthesis. You will study the architectural guidelines introduced by Radford et al. in the landmark DCGAN paper—mastering Transposed Convolutions, Batch Normalization dynamics, strided convolutions, non-linear activations (LeakyReLU and ReLU), and implementing a production-grade DCGAN in PyTorch for high-fidelity image generation on the CIFAR-10 / CelebA datasets.

---

## 📝 Core Technical Objectives
- **Transposed Convolutions & Spatial Upsampling:** Understanding spatial dimension expansion mechanics using `nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)`. Eliminating spatial pooling (MaxPool) in favor of fractional-strided convolutions to learn learnable spatial upsampling filters:

$$H{\text{out}} = (H{\text{in}} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\size}$$

- **Batch Normalization Dynamics in GANs:** Stabilizing adversarial training by normalizing feature map activations across mini-batches ($x \to \hat{x} \to \gamma \hat{x} + \beta$). Preventing internal covariate shift, avoiding mode collapse, and enabling deeper spatial network topologies without gradient vanishing/explosion.

- **DCGAN Architectural Conventions:**
  - **Generator $G$:** Utilizing Transposed Convolutions, Batch Normalization after every layer (except the final output layer), ReLU activations for hidden layers, and Tanh activation for the final image output layer ($[-1, 1]$ pixel range).
  - **Discriminator $D$:** Replacing pooling layers with strided `nn.Conv2d` layers for spatial downsampling, applying Batch Normalization (except on the raw input layer), LeakyReLU activations ($\alpha = 0.2$), and returning raw unnormalized logits for stable loss computation.

- **4D Spatial Tensor Management:** Transforming 1D latent noise vectors $z \in \mathbb{R}^{d_z}$ into 4D spatial feature maps $\mathbf{Z} \in \mathbb{R}^{N \times d_z \times 1 \times 1}$ using `z.view(len(z), z_dim, 1, 1)` to initiate the spatial upsampling cascade.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, PyTorch DCGAN programming assignments, and video generation labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Lecture%20Files/Course-1-Week-2.pdf)** | Formal visual reference covering Transposed Convolution stride arithmetic, Batch Normalization mean/variance scaling formulas, $G$ and $D$ activation function selections, and spatial feature map tracking. |
| **[DCGAN Assignment](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Assingment/C1_W2_Assignment.ipynb)** | Complete PyTorch DCGAN implementation: building `Generator` blocks using `nn.ConvTranspose2d` + `nn.BatchNorm2d` + `nn.ReLU`, building `Discriminator` blocks using `nn.Conv2d` + `nn.BatchNorm2d` + `nn.LeakyReLU`, custom weight initialization ($\mu=0, \sigma=0.02$), and training on multi-channel image datasets. |
| **[Video Generation Lab](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Assingment/C_1_2_Optional_Labs_1_Video_Generation.ipynb)** | Extending 2D spatial DCGAN concepts to 3D spatio-temporal representations using `nn.Conv3d` and `nn.ConvTranspose3d` layers to generate continuous synthetic video frame sequences. |

---

## 💡 Visual Pipeline Reference

The spatial dimension progression across $G$'s transposed convolution upsampling cascade alongside $D$'s strided downsampling pipeline:

- **Generator Transposed Convolution Spatial Cascade ($64 \times 64$ RGB Generation)**:

$$\begin{aligned}   z \in \mathbb{R}^{B \times d_z \times 1 \times 1} &\xrightarrow{\text{ConvTranspose2d}(k=4, s=1, p=0)} \mathbb{R}^{B \times 512 \times 4 \times 4} \xrightarrow{\text{BN } + \text{ ReLU}} \\   &\xrightarrow{\text{ConvTranspose2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 256 \times 8 \times 8} \xrightarrow{\text{BN } + \text{ ReLU}} \\   &\xrightarrow{\text{ConvTranspose2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 128 \times 16 \times 16} \xrightarrow{\text{BN } + \text{ ReLU}} \\   &\xrightarrow{\text{ConvTranspose2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 64 \times 32 \times 32} \xrightarrow{\text{BN } + \text{ ReLU}} \\   &\xrightarrow{\text{ConvTranspose2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 3 \times 64 \times 64} \xrightarrow{\text{Tanh}} \hat{X}_{\text{RGB}} \in [-1, 1]^{B \times 3 \times 64 \times 64}   \end{aligned}$$

- **Discriminator Strided Downsampling Convolutional Cascade**:

$$\begin{aligned}   X_{\text{RGB}} \in \mathbb{R}^{B \times 3 \times 64 \times 64} &\xrightarrow{\text{Conv2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 64 \times 32 \times 32} \xrightarrow{\text{LeakyReLU}(0.2)} \\   &\xrightarrow{\text{Conv2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 128 \times 16 \times 16} \xrightarrow{\text{BN } + \text{ LeakyReLU}(0.2)} \\   &\xrightarrow{\text{Conv2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 256 \times 8 \times 8} \xrightarrow{\text{BN } + \text{ LeakyReLU}(0.2)} \\   &\xrightarrow{\text{Conv2d}(k=4, s=2, p=1)} \mathbb{R}^{B \times 512 \times 4 \times 4} \xrightarrow{\text{BN } + \text{ LeakyReLU}(0.2)} \\   &\xrightarrow{\text{Conv2d}(k=4, s=1, p=0)} \mathbb{R}^{B \times 1 \times 1 \times 1} \xrightarrow{\text{Flatten}} \text{Logits } \in \mathbb{R}^{B \times 1}   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Spatial Feature Theory & Convolutional Mechanics
- **Fractional-Strided Arithmetic:** Deriving exact output spatial shapes $(H_{\text{out}}, W_{\text{out}})$ for upsampling layers to achieve pixel-perfect image synthesis without spatial boundary distortion.

- **Weight Initialization Protocol:** Implementing the DCGAN normal distribution weight initialization scheme ($\mathbf{W} \sim \mathcal{N}(0, 0.02)$, $\mathbf{b} = 0$) using custom PyTorch layer traversal functions (`net.apply(weights_init)`).

- **Grid Checkerboard Artifact Avoidance:** Understanding how kernel sizes and stride ratios in transposed convolutions can cause uneven overlapping receptive fields (checkerboard pattern artifacts) and selecting optimal $(k, s, p)$ combinations.


### 🤖 Applied PyTorch & Advanced GAN Architecture
- **Modular PyTorch Convolutional Blocks:** Engineering clean reusable building blocks (`nn.Sequential`) containing convolutional, normalization, and activation layers.

- **Image Normalization Alignment:** Matching dataset input transformations (`transforms.Normalize((0.5,), (0.5,))`) to $G$'s Tanh output distribution range $[-1, 1]$.

- **3D Video Spatio-Temporal Synthesis:** Extending 2D spatial feature convolutions to 3D time-series image volumes ($B \times C \times T \times H \times W$) for video sequence generation.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Computer Vision & Data Augmentation | Tensor Acceleration | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Transforms_&_DataLoaders-0056D2?style=flat&logo=python&logoColor=white) | ![CUDA](https://img.shields.io/badge/CUDA-GPU_Acceleration-76B900?style=flat&logo=nvidia&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
