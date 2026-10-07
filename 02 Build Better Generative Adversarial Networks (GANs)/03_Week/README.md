# 🎭 Week 3: StyleGAN & Advanced Generative Architectures

Welcome to Week 3 of **Build Better Generative Adversarial Networks (GANs)** (Course 2 of the **DeepLearning.AI GANs Specialization**)! This final module of Course 2 explores state-of-the-art generative architecture advancements, focusing on Kerras et al.'s landmark StyleGAN (and StyleGAN2) alongside Brock et al.'s BigGAN. You will disassemble the traditional generator pipeline and construct StyleGAN's core building blocks from scratch in PyTorch—mastering Mapping Networks ($z \to w$), Adaptive Instance Normalization (AdaIN), style modulation/demodulation, progressive growing mechanics, and per-pixel noise injection for fine stochastic details.

---

## 📝 Core Technical Objectives
- **8-Layer Mapping Network ($z \to w$ Disentanglement):** Mapping arbitrary Gaussian noise $z \in \mathbb{R}^{512}$ to an intermediate disentangled latent space $w \in \mathbb{W} \subset \mathbb{R}^{512}$ using an 8-layer fully-connected MLP. Disentangling latent attribute axes to align with linear visual features, overcoming dataset manifold curvature and non-linear feature correlation:

$$w = f_{\text{MLP}}(z)$$

- **Adaptive Instance Normalization (AdaIN):** Replacing direct noise tensor feeds with style-driven spatial feature modulation. Normalizing feature map channels $x_i$ to zero mean and unit variance, then scaling and shifting using affine style transformations $(\mathbf{y}_{s, i}, \mathbf{y}_{b, i})$ derived from intermediate latent vector $w$:

$$\text{AdaIN}(x_i, y) = \mathbf{y}_{s, i} \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + \mathbf{y}_{b, i}$$

- **Stochastic Variation via Per-Pixel Noise Injection:** Injecting scaled Gaussian noise maps $B \sim \mathcal{N}(0, I)$ directly into each convolution layer after feature normalization. Controlling fine-grained stochastic visual features (e.g., hair strands, freckles, skin pores, background textures) independently of high-level coarse style layout:

$$x_{\text{out}} = x_{\text{in}} + \mathbf{\gamma}_i \odot B_i$$

- **Progressive Growing & Constant Input Initialization:** Replacing random latent vectors as spatial inputs with a learned constant initial tensor $\mathbf{X}_{\text{const}} \in \mathbb{R}^{512 \times 4 \times 4}$. Dynamically increasing resolution through progressive training phases ($4 \times 4 \to 8 \times 8 \to \dots \to 1024 \times 1024$) using smooth fading skip-connections to ensure global structural stability.

- **StyleGAN2 & BigGAN Architectural Refinements:**
  - **StyleGAN2:** Eliminating AdaIN normalization artifacts by replacing explicit normalization with Weight Demodulation directly on convolution kernel weights $\mathbf{W}^{\prime}{i,j,k} = \mathbf{s}i \cdot \mathbf{W}{i,j,k}$ and $\mathbf{W}^{\prime\prime}{i,j,k} = \frac{\mathbf{W}^{\prime}{i,j,k}}{\sqrt{\sum{i,k} (\mathbf{W}^{\prime}{i,j,k})^2 + \epsilon}}$.
  - **BigGAN:** Scaling generator batch sizes ($2048$), channel capacities, residual blocks, class-conditional shared embeddings, and applying Self-Attention modules alongside the Truncation Trick.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, PyTorch StyleGAN assignment, and specialized advanced labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Lecture%20Files/Course-2-Week-3.pdf)** | Formal visual reference covering Mapping Network $z \to w$ geometry, AdaIN formula breakdowns, progressive resolution transition math ($\alpha$-fading), per-pixel noise injection mechanics, and weight demodulation algebra. |
| **[Components of StyleGAN Assignment](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Assingment/C2W3_Assignment.ipynb)** | Complete PyTorch implementation of StyleGAN building blocks: constructing the 8-layer MLP `MappingNetwork`, building `AdaIN` modules, creating custom learnable constant initial tensors, engineering stochastic per-pixel noise injection layers, and composing the complete Synthesis Network. |
| **[StyleGAN2 Lab](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Assingment/C2W3_Optional_Labs_1_StyleGAN2.ipynb)** | Implementing StyleGAN2 weight modulation and demodulation layers, path length regularization ($R_{\text{PL}}$), and main-path skip architecture additions to remove droplet normalization artifacts. |
| **[BigGAN Lab](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Assingment/C2W3_Optional_Labs_2_BigGAN.ipynb)** | Constructing BigGAN class-conditional ResNet blocks (`BigGANBlock`), hierarchical latent noise routing $z \to [z_0, z_1, \dots]$, spatial Self-Attention layers, and large-scale distributed training setups. |

---

## 💡 Visual Pipeline Reference

The complete StyleGAN generator network pass showcasing intermediate mapping $w$ alongside AdaIN and noise injection into synthesis blocks:

- **StyleGAN Synthesis Architecture Pipeline**:

$$\begin{aligned}   z \sim \mathcal{N}(0, I) \longrightarrow \mathbf{\text{Mapping Network}}_{\text{MLP} \times 8} \longrightarrow w \in \mathbb{W} \longrightarrow &\begin{cases} \text{Affine Transformation } (\mathbf{y}_{s}, \mathbf{y}_{b}) \\ \text{Fed to AdaIN at every resolution block} \end{cases} \\   \mathbf{X}_{\text{const}} \in \mathbb{R}^{512 \times 4 \times 4} \longrightarrow \left[ \mathbf{\text{Conv } 3 \times 3} \right] &\xrightarrow{+ \text{ Noise } B_1} \text{AdaIN}(w) \longrightarrow \dots \xrightarrow{\text{Upsample}} \mathbf{\text{Conv } 3 \times 3} \xrightarrow{+ \text{ Noise } B_n} \text{AdaIN}(w) \longrightarrow \hat{X}_{\text{RGB}}   \end{aligned}$$

- **StyleGAN Architectural Layers & Functional Responsibilities**:

| Sub-Module / Layer | Primary Input Tensor | Structural Function / Output | Visual Control Tier |
| :--- | :--- | :--- | :--- |
| **Mapping Network** | $z \in \mathbb{R}^{512} \sim \mathcal{N}(0, I)$ | Unrolls $z$ into disentangled space $w \in \mathbb{R}^{512}$ | Global Manifold Linearization |
| **Learned Constant Input** | Parameter $\mathbf{X}_{\text{const}} \in \mathbb{R}^{512 \times 4 \times 4}$ | Serves as static spatial seed tensor for synthesis | Fixes Global Geometry Base |
| **AdaIN / ModDemod Layer** | Feature Map $x_i$ & Style $w$ | Scales/shifts channels based on intermediate vector $w$ | Coarse/Medium Style Attributes |
| **Noise Injection Module**| Spatial Noise $B \in \mathbb{R}^{1 \times H \times W}$ | Multiplies learnable scale $\gamma_i$ and adds to features | Fine Stochastic Details (Hair/Pores) |
| **ToRGB Block** | Feature Map $\mathbb{R}^{C \times H \times W}$ | Converts latent features to 3-channel RGB image ($[-1, 1]$) | Final Resolution Visual Synthesis |

---

## 🎯 Technical Skills Architecture

### 📊 Disentangled Generative Theory & Style Control
- **Intermediate Latent Space Geometry ($w$-Space):** Explaining why mapping $z \to w$ eliminates feature entanglement, allowing continuous linear attribute adjustments without unintended secondary visual changes.

- **Hierarchical Style Transfer Mechanics:** Understanding how feeding intermediate vector $w$ at different resolution scales controls coarse aspects ($4 \times 4 - 8 \times 8$: pose, face shape), medium aspects ($16 \times 16 - 32 \times 32$: facial features, hairstyle), and fine aspects ($64 \times 64 - 1024 \times 1024$: color scheme, micro-textures).

- **Weight Demodulation Physics:** Deriving how StyleGAN2 combines convolution weights and style scale factors into a single normalized tensor $\mathbf{W}^{\prime\prime}$ to accelerate inference and eliminate normalization spatial artifacts.


### 🤖 Applied PyTorch & Advanced Synthesis Engineering
- **Custom PyTorch StyleGAN Submodules:** Constructing dynamic PyTorch classes (`MappingNetwork`, `AdaIN`, `StyleGANGenerator`, `NoiseInjection`) with parameter registration (`nn.Parameter`) for constant input vectors.

- **Dynamic Broadcasting & Tensor Normalization:** Writing vectorized channel-wise mean and variance calculations across spatial dimensions (`x.mean([2, 3], keepdim=True)`) to execute instanced feature normalization.

- **Weight Demodulation Convolution Implementations:** Fusing style modulation vectors and standard convolution weights into grouped 2D convolutions (`nn.functional.conv2d`) in PyTorch.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Generative Architectures | Advanced Optimizations | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![StyleGAN](https://img.shields.io/badge/GAN-StyleGAN1_--_3_&_BigGAN-0056D2?style=flat&logo=python&logoColor=white) | ![CUDA](https://img.shields.io/badge/CUDA-Mixed_Precision_&_Custom_Ops-76B900?style=flat&logo=nvidia&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
