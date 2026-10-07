# 🎛️ Week 4: Conditional GANs & Controllable Generation

Welcome to Week 4 of **Build Basic Generative Adversarial Networks (GANs)** (Course 1 of the **DeepLearning.AI GANs Specialization**)! This final module of Course 1 transitions from unconstrained, random generation to precise feature control and conditional synthesis. You will master two distinct paradigms of controlled generation: explicit class conditioning via Conditional GANs (cGAN) and post-hoc latent space feature manipulation using direction vectors, classifier gradients, and disentangled latent spaces (InfoGAN).

---

## header Core Technical Objectives
- **Explicit Class Conditioning (cGAN):** Conditioning both the Generator $G(z, y)$ and Discriminator $D(x, y)$ on categorical class labels $y$. Concatenating one-hot encoded label vectors or dense class embedding tensors directly to input noise vectors $z$ and spatial image features $x$:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x, y \sim p_{\text{data}}(x, y)}[\log D(x, y)] + \mathbb{E}_{z \sim p_z(z), y \sim p_y(y)}[\log(1 - D(G(z, y), y))]$$

- **One-Hot Encoding & Spatial Label Tiling:** Transforming target class integers $y \in \{0, \dots, C-1\}$ into dynamic one-hot vectors $\mathbf{Y} \in \mathbb{R}^{B \times C}$. Expanding one-hot vectors to match spatial image dimensions $\mathbb{R}^{B \times C \times H \times W}$ via tensor repetition/tiling to concatenate across channel dimensions: `torch.cat([x, y_spatial], dim=1)`.

- **Latent Space Vector Algebra:** Manipulating latent noise vectors $z \in \mathbb{R}^{dz}$ along semantic direction vectors $v{\text{feature}}$. Extracting directional vectors by computing differences between mean latent encodings of target feature groups:

$$v{\text{feature}} = \mu\left(z{\text{feature\present}}\right) - \mu\left(z{\text{feature\absent}}\right) \quad \implies \quad z{\text{new}} = z{\text{original}} + \alpha \cdot v{\text{feature}}$$

- **Classifier-Guided Feature Traversal:** Guiding latent vector updates using a auxiliary pre-trained feature classifier $C_{\text{aux}}$. Updating $z$ via gradient ascent to increase target feature probability $P(\text{feature} \mid G(z))$ without retraining $G$:

$$z^{(t+1)} = z^{(t)} + \gamma \cdot \nabla_{z} \mathcal{L}_{\text{class}}\left(C_{\text{aux}}(G(z)), y_{\text{target}}\right)$$

- **Disentangled Representation Mechanics (InfoGAN):** Maximizing the mutual information $I(c; G(z, c))$ between a structured latent code $c$ and the generated output to isolate continuous latent attributes (e.g., rotation, width, thickness) without explicit label supervision.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, PyTorch conditional assignments, and latent manipulation labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/04_Week/Lecture%20Files/Course-1-Week-4.pdf)** | Formal visual reference covering one-hot label concatenation geometry, spatial tiling math, latent space direction vector arithmetic, classifier gradient backpropagation to $z$, and mutual information lower bounds. |
| **[Build a Conditional GAN](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/04_Week/Assingment/C1W4A_Build_a_Conditional_GAN.ipynb)** | Complete PyTorch implementation of cGAN on MNIST: implementing dynamic one-hot encoding (`F.one_hot`), concatenating labels to latent noise vectors $z$ in $G$, spatially tiling labels across channel dimensions for $D$, and generating specific user-defined digits. |
| **[Controllable Generation](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/04_Week/Assingment/C1W4B_Controllable_Generation.ipynb)** | Post-hoc latent space manipulation on CelebA: calculating mean feature directions $v_{\text{feature}}$ for facial attributes (e.g., smiling, eyeglasses, age), steering latent noise vectors $z$, and utilizing pre-trained classifier gradients to continuously adjust feature intensities. |
| **[InfoGAN Lab](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/04_Week/Assingment/C1W4B_Optional_Labs_1_InfoGAN.ipynb)** | Unsupervised feature disentanglement: training an auxiliary head $Q(c \mid x)$ to approximate the posterior distribution of latent codes $c$ and learning disentangled continuous style variations without explicit attribute labels. |

---

## 💡 Visual Pipeline Reference

The Conditional GAN tensor concatenation pass alongside the classifier-guided latent vector optimization loop:

- **Conditional GAN Tensor Concatenation Architecture**:

$$\begin{aligned}   \text{Generator: } &\left[ z \in \mathbb{R}^{B \times d_z}, y_{\text{onehot}} \in \mathbb{R}^{B \times C} \right] \xrightarrow{\text{torch.cat}} \mathbb{R}^{B \times (d_z + C)} \longrightarrow \mathbf{G}_{\theta} \longrightarrow \hat{X}_{\text{fake}} \in \mathbb{R}^{B \times 1 \times 28 \times 28} \\   \text{Discriminator: } &\begin{cases} X \in \mathbb{R}^{B \times 1 \times 28 \times 28} \\ y_{\text{spatial}} \in \mathbb{R}^{B \times C \times 28 \times 28} \end{cases} \xrightarrow{\text{torch.cat}} \mathbb{R}^{B \times (1 + C) \times 28 \times 28} \longrightarrow \mathbf{D}_{\phi} \longrightarrow \text{Logits} \in \mathbb{R}^{B \times 1}   \end{aligned}$$

- **Classifier Gradient Latent Traversal ($z$-Optimization)**:

$$z_0 \sim \mathcal{N}(0, I) \longrightarrow \mathbf{G}(z_t) \longrightarrow \mathbf{C}_{\text{aux}}(\mathbf{G}(z_t)) \longrightarrow \mathcal{L}_{\text{target}} \xrightarrow{\text{autograd.grad}} \nabla_{z} \mathcal{L} \implies z_{t+1} = z_t + \gamma \nabla_z \mathcal{L}$$

---

## 🎯 Technical Skills Architecture

### 📊 Conditional Theory & Latent Space Geometry
- **Tensor Concatenation Dynamics:** Proving why conditioning requires feeding target labels to *both* $G$ and $D$, preventing $G$ from ignoring conditions and preventing $D$ from evaluating authenticity independent of target class.

- **Disentangled Latent Spaces:** Distinguishing between entangled representations (where changing one latent component unpredictably alters multiple visual traits) and disentangled representations (where orthogonal axes isolate single independent attributes like pose or illumination).

- **Classifier Gradient Traversal Physics:** Understanding how backpropagating loss through fixed $C_{\text{aux}}$ weights down to $z$ allows smooth trajectory navigation across $G$'s learned manifold.


### 🤖 Applied PyTorch & Controllable Synthesis
- **Dynamic Spatial Tiling Operations:** Converting 1D categorical labels into 4D spatial tensors using PyTorch `repeat` and `expand` functions: `y_onehot.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)`.

- **Vectorized Direction Extraction:** Computing normalized direction vectors $v = \frac{\bar{z}_1 - \bar{z}_0}{\Vert{}\bar{z}_1 - \bar{z}_0\Vert{}_2}$ across latent batches and interpolating along line segments: $z(\alpha) = z_0 + \alpha v$.

- **PyTorch Gradient Retain Mechanics:** Setting `z.requires_grad_(True)` and executing optimizer steps directly on latent vectors $z$ while holding generator weights $\theta_G$ strictly frozen.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Tensor Manipulation & One-Hot | Auxiliary Classifiers | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torch Functional](https://img.shields.io/badge/PyTorch-nn.functional_&_OneHot-0056D2?style=flat&logo=python&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Pretrained_Classifiers-76B900?style=flat&logo=pytorch&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
