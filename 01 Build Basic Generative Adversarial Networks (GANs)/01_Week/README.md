# 🎨 Week 1: Introduction to Generative Adversarial Networks (GANs)

Welcome to Week 1 of **Build Basic Generative Adversarial Networks (GANs)** (Course 1 of the **DeepLearning.AI GANs Specialization**)! This module establishes the core adversarial framework that powers modern generative AI. You will study the zero-sum game dynamics between two neural networks—the Generator and the Discriminator—and implement your very first fully functional Deep Convolutional/Fully Connected GAN in PyTorch from scratch to synthesize realistic handwritten digits on the MNIST dataset.

---

## 📝 Core Technical Objectives
- **Adversarial Game Theory Mechanics:** Formulating GAN training as a two-player zero-sum minimax game between a Generator $G_{\theta}$ attempting to map noise vectors $z \sim p_z$ into data distribution space $p_{\text{data}}$, and a Discriminator $D_{\phi}$ estimating the scalar probability that a given image originates from real data rather than $G$:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- **Binary Cross-Entropy (BCE) Loss Optimization:** Optimizing real and fake output predictions using PyTorch's `nn.BCEWithLogitsLoss()`. Training $D$ to maximize real classification accuracy while simultaneously training $G$ using inverted target labels ($1$ instead of $0$) to fool $D$:

$$\mathcal{L}_{D} = -\frac{1}{2}\left[ \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] \right], \quad \mathcal{L}_{G} = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

- **Generator & Discriminator Architecture Design:** Constructing fully-connected and convolutional neural network blocks with Spectral/Batch Normalization, LeakyReLU non-linearities ($\alpha = 0.2$) for $D$ to maintain non-zero gradients, and Sigmoid/Tanh activations for raw spatial image generation.

- **PyTorch Tensor Workflows & GPU Acceleration:** Managing PyTorch tensors (`torch.Tensor`), automatic gradient tracking (`loss.backward()`), explicit optimizer stepping (`optimizer.step()`, `optimizer.zero_grad()`), parameter isolation using `.detach()`, and GPU tensor migration (`.to(device)`).

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, PyTorch tutorial notebooks, and GAN programming assignments are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/01_Week/Lecture%20Files/Course-1-Week-1.pdf)** | Formal visual reference covering adversarial intuition, discriminator vs. generator loss curves, minimax payoff functions, real-world GAN applications (StyleGAN, BigGAN), and PyTorch autograd dynamics. |
| **[Intro to PyTorch](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/01_Week/01%20Assingment/Course-1-Week-1-Optional-Labs-1-Intro-to-PyTorch.ipynb)** | Hands-on PyTorch primer: creating tensors, understanding dynamic computation graphs (`autograd`), implementing linear layers, activation functions, custom loss functions, and running training loops using `torch.optim.Adam`. |
| **[Pre-trained GAN Exploration](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/01_Week/01%20Assingment/Course-1-Week-1-Optional-Labs-2-Inputs-to-a-pre-trained-GAN.ipynb)** | Sampling latent noise vectors $z \in \mathbb{R}^{d_{\text{z}}}$, observing continuous latent space interpolation, and inspecting visual feature transformations in pre-trained generative networks. |
| **[Your First GAN Assignment](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/01_Week/01%20Assingment/Course-1-Week-1-Practice-Labs-1-Your-First-GAN.ipynb)** | Ground-up PyTorch implementation: building custom `Generator` and `Discriminator` classes, writing latent noise generator functions (`get_noise`), defining alternating training loops, and generating synthetic MNIST digits. |

---

## 💡 Visual Pipeline Reference

The complete adversarial feedback loop alongside the tensor computation pass:

- **Adversarial Training Dataflow & Gradient Flow**:

  $$\begin{aligned}   \text{Noise Vector } z \sim \mathcal{N}(0, I) &\longrightarrow \mathbf{G}_{\theta}(z) \longrightarrow \text{Fake Image } \hat{x} \longrightarrow \mathbf{D}_{\phi}(\hat{x}) \longrightarrow \text{Loss } \mathcal{L}_G \implies \nabla_{\theta} \text{ updates } G \\   \text{Real Image } x \sim p_{\text{data}} &\longrightarrow \mathbf{D}_{\phi}(x) \longrightarrow \text{Loss } \mathcal{L}_D(\text{Real}, \text{Fake.detach()}) \implies \nabla_{\phi} \text{ updates } D   \end{aligned}$$

- **Generator vs. Discriminator Functional Pipeline**:

$$\begin{aligned}   G(z): \mathbb{R}^{z\_dim} &\xrightarrow{\text{Linear} \to \text{BatchNorm} \to \text{ReLU}} \mathbb{R}^{256} \xrightarrow{\dots} \mathbb{R}^{784} \xrightarrow{\text{Sigmoid}} \hat{X}_{\text{flat}} \\   D(x): \mathbb{R}^{784} &\xrightarrow{\text{Linear} \to \text{LeakyReLU}(0.2)} \mathbb{R}^{512} \xrightarrow{\dots} \mathbb{R}^{1} \xrightarrow{\text{Logits}} \text{Prediction Probability}   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Adversarial Theory & Mathematical Optimization
- **Zero-Sum Dynamics & Nash Equilibrium:** Explaining why $G$ and $D$ reach Nash Equilibrium when $p_g = p_{\text{data}}$ and $D(x) = \frac{1}{2}$ across all inputs.

- **Discriminator Gradient Stabilization:** Utilizing `LeakyReLU` activations in $D$ to prevent sparse/dead gradients that halt $G$'s early learning phase.

- **Gradient Detaching Mechanics:** Detaching generated tensors (`fake.detach()`) during the Discriminator pass to prevent compute graph leakage and backward gradient flow into $G$ during $D$'s update.


### 🤖 Applied PyTorch & Generative Modeling
- **Custom PyTorch Model Construction:** Building modular neural network layers using `torch.nn.Sequential`, `nn.Linear`, `nn.BatchNorm1d`, and custom initializations.

- **Alternating Optimization Loops:** Managing separate `torch.optim.Adam` optimizers for $G$ and $D$, ensuring clear step-by-step weight updates without state corruption.

- **Latent Space Sampling:** Constructing reproducible standard normal distribution samplers (`torch.randn`) to steer generated visual manifestations.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Computer Vision & Plotting | Tensor Acceleration | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Transforms_&_Datasets-0056D2?style=flat&logo=python&logoColor=white) | ![CUDA](https://img.shields.io/badge/CUDA-GPU_Acceleration-76B900?style=flat&logo=nvidia&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
