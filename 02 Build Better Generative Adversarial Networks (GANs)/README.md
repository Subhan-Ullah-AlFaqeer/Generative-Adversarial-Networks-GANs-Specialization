# 🚀 Course 2: Build Better Generative Adversarial Networks (GANs)

Welcome to **Course 2** of the **DeepLearning.AI Generative Adversarial Networks (GANs) Specialization**! This course bridges foundational adversarial learning and state-of-the-art generative architecture design. You will transition from basic GAN training to implementing quantitative evaluation metrics, auditing machine learning bias, exploring alternative generative paradigms (VAEs, Score-Based Models, and NeRF), and constructing Kerras et al.'s landmark StyleGAN architecture from scratch in PyTorch.

Across 3 comprehensive modules, you will build production-grade evaluation pipelines and state-of-the-art generative blocks—computing Fréchet Inception Distance (FID) via deep Inception-v3 embeddings, performing audit-level classification to identify algorithmic bias across demographic groups, implementing Variational Autoencoders (VAEs) via the reparameterization trick, and engineering disentangled latent mapping networks ($z \to w$), Adaptive Instance Normalization (AdaIN), and stochastic noise injection layers.

---

## 📝 Core Technical Objectives
- **Quantitative Generative Assessment & FID Mechanics:** Overcoming pixel-space distance limitations by extracting high-level $2048$-dimensional feature activations through pre-trained Inception-v3 networks. Evaluating visual fidelity and distribution coverage using Fréchet Inception Distance (FID) between multivariate Gaussian distributions $\mathcal{N}(\mu_r, \Sigma_r)$ and $\mathcal{N}(\mu_g, \Sigma_g)$:

$$\text{FID} = \Vert{}\mu_r - \mu_g\Vert{}_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

- **Generative Trilemma & Alternative Paradigms:** Navigating performance trade-offs across generative architectures (GANs vs. VAEs vs. Score-Based Diffusion Models vs. NeRF). Optimizing Variational Autoencoders via the Evidence Lower Bound (ELBO) and analytical KL divergence using the Reparameterization Trick ($z = \mu + \sigma \odot \epsilon$):

$$\mathcal{L}_{\text{ELBO}}(\theta, \phi; x) = \mathbb{E}_{q_{\phi}(z\vert{}x)}[\log p_{\theta}(x\vert{}z)] - D_{\text{KL}}\left(q_{\phi}(z\vert{}x) \parallel p(z)\right)$$

- **Algorithmic Bias, Fairness & Feature Auditing:** Identifying dataset representation skews, defining formal mathematical fairness constraints (Demographic Parity, Equalized Odds), and building diagnostic auditing pipelines to evaluate class representation and attribute distributions across protected demographic subgroups.

- **StyleGAN Architecture & Disentangled Latent Control:** Constructing StyleGAN building blocks: unrolling Gaussian noise $z$ into a disentangled intermediate space $w \in \mathbb{W}$ via an 8-layer MLP mapping network, modulating feature channel scale/bias using Adaptive Instance Normalization (AdaIN), injecting per-pixel Gaussian noise for fine stochastic details, and implementing weight demodulation layers (StyleGAN2).

---

## 🧪 Course Architecture & Module Matrix

This course is structured into 3 sequential modules covering quantitative evaluation, alternative model paradigms & bias auditing, and state-of-the-art StyleGAN architectural advancements:

| Module / Directory | Operational Focus & Technical Topics | Key Notebooks & Deliverables |
| :--- | :--- | :--- |
| **[01_Week](./01_Week)** | **Evaluation of GANs**: Inception-v3 deep feature extraction, Inception Score (IS), Fréchet Inception Distance (FID) matrix math, Precision vs. Recall for generative models, Truncation Trick sampling, and Perceptual Path Length (PPL) latent smoothness. | • `C2W1_Assignment.ipynb`<br>• `C2W1_Optional-Labs_PPL.ipynb` |
| **[02_Week](./02_Week)** | **GAN Disadvantages, VAEs, Bias & NeRF**: The Generative Trilemma, Variational Autoencoders (VAEs), Reparameterization Trick, Score-Based Generative Models / Langevin dynamics, sources of dataset bias, demographic fairness metrics, and Neural Radiance Fields (NeRF). | • `C2W2_Assignment.ipynb`<br>• `C2W2_Optional_Labs_2_Score_Based_Generative_Modeling.ipynb`<br>• `C2W2_Optional_Labs_3_GAN_Debiasing.ipynb`<br>• `C2W2_Optional_Labs_4_Neural_Radiance_Fields_(NeRF).ipynb` |
| **[03_Week](./03_Week)** | **StyleGAN & Advanced Architectures**: 8-Layer Mapping Networks ($z \to w$), Adaptive Instance Normalization (AdaIN), learned constant inputs $\mathbf{X}_{\text{const}}$, per-pixel stochastic noise injection, progressive resolution growing, StyleGAN2 Weight Demodulation, and BigGAN conditional ResNet blocks. | • `C2W3_Assignment.ipynb`<br>• `C2W3_Optional_Labs_1_StyleGAN2.ipynb`<br>• `C2W3_Optional_Labs_2_BigGAN.ipynb` |

---

## 💡 Visual Pipeline Reference

The unified diagnostic equations and mathematical foundations governing Course 2:

- **Fréchet Inception Distance (Wasserstein-2 Metric in Feature Space)**:

$$\text{FID} = \Vert{}\mu_r - \mu_g\Vert{}_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

- **Variational Autoencoder (VAE) ELBO Loss & Gaussian KL Divergence**:

$$\mathcal{L}_{\text{VAE}} = \text{Reconstruction\Loss}(x, \hat{x}) - \frac{1}{2} \sum_{j=1}^{d_z} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

- **Adaptive Instance Normalization (AdaIN) Channel Transformation**:

$$\text{AdaIN}(x_i, y) = \mathbf{y}_{s, i} \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + \mathbf{y}_{b, i}$$

- **StyleGAN2 Weight Demodulation Formula**:

$$\mathbf{W}^{\prime\prime}_{i,j,k} = \frac{\mathbf{s}_i \cdot \mathbf{W}_{i,j,k}}{\sqrt{\sum_{i,k} (\mathbf{s}_i \cdot \mathbf{W}_{i,j,k})^2 + \epsilon}}$$

---

## 🎯 Technical Skills Architecture

### 📊 Generative Theory & Advanced Evaluation
- **Statistical Metric Calculus:** Computing matrix square roots ($\sqrt{\Sigma_r \Sigma_g}$) and multivariate Gaussian stats over feature embeddings, proving why FID correlates tightly with perceptual visual human assessment.

- **Latent Disentanglement & Manifold Linearization:** Proving how mapping $z \to w$ unrolls curved latent manifolds into linear orthogonal attribute axes, allowing precise visual editing without unwanted feature entanglements.

- **Generative Trilemma & Fairness Mechanics:** Evaluating generative models across quality, coverage, and speed trade-offs, alongside implementing statistical demographic parity constraints to mitigate bias in synthetic output distributions.


### 🤖 Applied PyTorch & Advanced Generative Engineering
- **Feature Extraction & Hooks:** Disabling final classifier layers in pre-trained `torchvision` models (`Inception-v3`, `VGG`) to extract intermediate activation tensors without recording autograd gradient graphs (`torch.no_grad()`).

- **Custom VAE & Reparameterization:** Implementing stochastic sampling layers ($z = \mu + \sigma \odot \epsilon$) to allow backpropagation through stochastic encoder outputs in PyTorch.

- **Modular StyleGAN Generator Engineering:** Constructing custom PyTorch modules (`MappingNetwork`, `AdaIN`, `NoiseInjection`, `StyleGANGenerator`) with parameter registration (`nn.Parameter`) for static baseline seeds and style modulation matrices.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Evaluation & Metrics Models | Generative & NeRF Libraries | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Inception_v3_&_VGG-0056D2?style=flat&logo=python&logoColor=white) | ![SciPy](https://img.shields.io/badge/SciPy-Matrix_Algebra_&_sqrtm-8CAAE6?style=flat&logo=scipy&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
