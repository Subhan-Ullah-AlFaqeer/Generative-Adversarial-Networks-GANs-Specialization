# 🎨 Course 1: Build Basic Generative Adversarial Networks (GANs)

Welcome to **Course 1** of the **DeepLearning.AI Generative Adversarial Networks (GANs) Specialization**! This course provides a comprehensive hands-on foundation in generative modeling—taking you from fundamental minimax game theory and PyTorch autograd dynamics to production-grade Deep Convolutional GANs (DCGAN), 1-Lipschitz enforcement via Wasserstein GANs with Gradient Penalty (WGAN-GP), and controllable synthesis using Conditional GANs (cGAN) and latent space manipulation.

Across 4 intensive modules, you will construct and train generative neural networks entirely from scratch in PyTorch—implementing adversarial loss functions, custom weight initializations, transposed convolutions, gradient penalty autograd loops, dynamic spatial label tiling, and vector arithmetic in latent $z$-space.

---

## 📝 Core Technical Objectives
- **Adversarial Framework & Minimax Game Theory:** Formulating GAN training as a two-player zero-sum game between a Generator $G(z)$ and a Discriminator $D(x)$, optimizing Binary Cross-Entropy (BCE) loss functions, and establishing Nash Equilibrium conditions ($p_g = p_{\text{data}}, D(x) = \frac{1}{2}$).

- **Spatial Upsampling & DCGAN Engineering:** Replacing fully connected networks with Deep Convolutional GANs (DCGAN), mastering Transposed Convolutions (`nn.ConvTranspose2d`), managing Batch Normalization dynamics, and applying specific non-linear activations (LeakyReLU for $D$, Tanh for $G$).

- **Wasserstein Distance & 1-Lipschitz Continuity:** Mitigating mode collapse and vanishing gradients by replacing JS/KL divergence with Earth Mover's Distance. Enforcing 1-Lipschitz continuity on the Critic network using Gradient Penalty ($\lambda = 10$) calculated over random linear image interpolations $\hat{x} = \epsilon x + (1 - \epsilon) G(z)$.

- **Conditional Synthesis & Latent Space Control:** Constructing Conditional GANs (cGAN) by concatenating one-hot target class labels $y$ to noise vectors $z$ and spatially tiling labels across image feature channels. Manipulating latent vectors $z$ via semantic direction vectors $v_{\text{feature}}$ and classifier gradients to continuously control visual outputs.

---

## 🧪 Course Architecture & Module Matrix

This course is structured into 4 sequential modules covering basic generative components, spatial convolutional architectures, Lipschitz-constrained stabilization, and controllable conditional synthesis:

| Module / Directory | Operational Focus & Technical Topics | Key Notebooks & Deliverables |
| :--- | :--- | :--- |
| **[01_Week](./01_Week)** | **Intro to GANs & PyTorch Basics**: Generative vs. Discriminative models, minimax objective function, BCE cost function, PyTorch tensor manipulation, dynamic computation graphs (`autograd`), and building a fully connected GAN for MNIST. | • `Course-1-Week-1-Optional-Labs-1-Intro-to-PyTorch.ipynb`<br>• `Course-1-Week-1-Optional-Labs-2-Inputs-to-a-pre-trained-GAN.ipynb`<br>• `Course-1-Week-1-Practice-Labs-1-Your-First-GAN.ipynb` |
| **[02_Week](./02_Week)** | **Deep Convolutional GANs (DCGAN)**: Transposed Convolutions (`nn.ConvTranspose2d`), spatial upsampling arithmetic, strided downsampling convolutions, Batch Normalization in GANs, custom normal weight initialization, and 3D spatio-temporal video generation. | • `C1_W2_Assignment.ipynb`<br>• `C_1_2_Optional_Labs_1_Video_Generation.ipynb` |
| **[03_Week](./03_Week)** | **Wasserstein GANs with Gradient Penalty (WGAN-GP)**: Mode collapse analysis, Earth Mover's Distance, Wasserstein loss, Critic networks, 1-Lipschitz continuity enforcement, higher-order autograd Gradient Penalty (`WGAN-GP`), and Spectral Normalization (`SN-GAN`). | • `C1W3_WGAN_GP.ipynb`<br>• `C1W3_SNGAN.ipynb`<br>• `C1W3_Optional-Labs-2-ProteinGAN.ipynb` |
| **[04_Week](./04_Week)** | **Conditional GANs & Controllable Generation**: Explicit class conditioning (cGAN), dynamic one-hot encoding, 4D spatial label tiling, latent vector arithmetic ($z$-space algebra), classifier-guided feature traversal, and unsupervised disentanglement (`InfoGAN`). | • `C1W4A_Build_a_Conditional_GAN.ipynb`<br>• `C1W4B_Controllable_Generation.ipynb`<br>• `C1W4B_Optional_Labs_1_InfoGAN.ipynb` |

---

## 💡 Visual Pipeline Reference

The unified diagnostic equations and mathematical foundations governing Course 1:

- **Full Adversarial Minimax Objective & BCE Loss**:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- **Transposed Convolution Spatial Dimension Scaling Formula**:

$$H{\text{out}} = (H{\text{in}} - 1) \times \text{stride} - 2 \times \text{padding} + \text{kernel\size}$$

- **Wasserstein Loss with 1-Lipschitz Gradient Penalty Constraint**:

$$\mathcal{L}{\text{Critic}} = \underset{\hat{x} \sim Pg}{\mathbb{E}}[C(G(z))] - \underset{x \sim Pr}{\mathbb{E}}[C(x)] + \lambda \underset{\hat{x} \sim P{\text{penalty}}}{\mathbb{E}} \left[ \left( \Vert{}\nabla_{\hat{x}} C(\hat{x})\Vert{}_2 - 1 \right)^2 \right]$$

- **Conditional GAN Tensor Concatenation Mechanics**:

$$G_{\text{input}} = \text{Concat}\left(z \in \mathbb{R}^{B \times d_z}, y_{\text{onehot}} \in \mathbb{R}^{B \times C}\right), \quad D_{\text{input}} = \text{Concat}\left(X \in \mathbb{R}^{B \times 1 \times H \times W}, Y_{\text{spatial}} \in \mathbb{R}^{B \times C \times H \times W}\right)$$

---

## 🎯 Technical Skills Architecture

### 📊 Generative Theory & Mathematical Optimization
- **Measure Theory & Distance Metrics:** Comparing Jensen-Shannon / KL divergences against Earth Mover's Distance, proving why Wasserstein metrics prevent vanishing gradients when real and generated distributions have disjoint supports.

- **Lipschitz Continuity Constraints:** Formulating 1-Lipschitz norm conditions ($\Vert{}\nabla f(x)\Vert{}_2 \le 1$) and implementing higher-order gradient penalties or spectral normalization to maintain bounded Critic slopes.

- **Latent Space Manifold Navigation:** Extracting semantic direction vectors $v = \mu(z_1) - \mu(z_0)$ to execute linear vector additions, interpolations, and classifier-guided gradient traversal in continuous latent spaces.


### 🤖 Applied PyTorch & Generative Model Engineering
- **PyTorch Model & Autograd Mastery:** Building modular convolutional neural networks (`nn.Sequential`, `nn.Module`), utilizing custom layer weight initializations (`apply(weights_init)`), and computing higher-order tensor gradients via `torch.autograd.grad`.

- **Asymmetrical Training Schedules:** Implementing multi-step optimization loops where the Critic updates $n_{\text{critic}} = 5$ times per single Generator update to maintain optimal dual function evaluation.

- **Dynamic Spatial Tensor Manipulation:** Engineering 4D spatial tensors, performing one-hot label encoding (`F.one_hot`), and tiling target attributes across feature map channels using `unsqueeze`, `expand`, and `repeat`.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Tensor Calculation & Autograd | Vision & Data Augmentation | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Autograd](https://img.shields.io/badge/PyTorch-Autograd_&_Functional-0056D2?style=flat&logo=python&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Transforms_&_DataLoaders-76B900?style=flat&logo=pytorch&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
