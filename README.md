# 🎨 DeepLearning.AI Generative Adversarial Networks (GANs) Specialization

Welcome to the root repository for the **DeepLearning.AI Generative Adversarial Networks (GANs) Specialization**, instructed by Sharon Zhou, Eda Zhou, Eric Zelikman, and Andrew Ng. This comprehensive 3-course series provides a production-grade journey through deep generative modeling—progressing from foundational game-theoretic adversarial objectives to state-of-the-art architectures, quantitative evaluation metrics, algorithmic fairness, and domain-specific applications in computer vision and privacy.

Throughout this specialization, you will construct, train, evaluate, and deploy advanced generative models using **PyTorch**. The curriculum bridges theoretical mathematical foundations with hands-on implementation—covering Deep Convolutional GANs (DCGAN), Wasserstein GANs with Gradient Penalty (WGAN-GP), Conditional/Controllable GANs, Fréchet Inception Distance (FID), Variational Autoencoders (VAEs), StyleGAN1–3, BigGAN, Pix2Pix paired translation, and CycleGAN unpaired style transfer.

---

## 📝 Specialization Overview & Technical Core
- **Game-Theoretic Adversarial Foundations:** Formulating generative learning as a zero-sum minimax game between a Generator $G_{\theta}$ and a Discriminator $D_{\phi}$. Transitioning from original Jensen-Shannon divergence min-max formulations to Earth Mover's (Wasserstein-1) Distance with Lipschitz continuity constraints enforced via Gradient Penalty:

  $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))] - \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$

- **Quantitative Generative Evaluation & Embeddings:** Assessing sample fidelity and distribution coverage in deep feature spaces using pre-trained Inception-v3 activation vectors. Calculating Fréchet Inception Distance (FID) over multivariate Gaussian distributions $\mathcal{N}(\mu_r, \Sigma_r)$ and $\mathcal{N}(\mu_g, \Sigma_g)$:

  $$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

- **State-of-the-Art Style Disentanglement & Control:** Unrolling Gaussian latent space $z \in \mathbb{R}^{512}$ into intermediate disentangled space $w \in \mathbb{W}$ via 8-layer MLPs in StyleGAN. Modulating channel scale and bias via Adaptive Instance Normalization (AdaIN) and weight demodulation while injecting per-pixel stochastic noise for fine visual details:

  $$\text{AdaIN}(x_i, y) = \mathbf{y}_{s, i} \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + \mathbf{y}_{b, i}$$

- **Paired & Unpaired Image-to-Image Translation:** Mapping visual domains using U-Net skip connection generators and $70 \times 70$ PatchGAN local discriminators (Pix2Pix), alongside dual-generator bi-directional style transfer constrained by Cycle Consistency and Identity losses (CycleGAN):

  $$\mathcal{L}_{\text{cyc}} = \mathbb{E}_{x}\left[ \|G_{Y \to X}(G_{X \to Y}(x)) - x\|_1 \right] + \mathbb{E}_{y}\left[ \|G_{X \to Y}(G_{Y \to X}(y)) - y\|_1 \right]$$

- **Social Implications, Bias & Privacy:** Auditing training datasets and pre-trained generators for demographic bias, defining formal fairness criteria, preserving privacy via differential privacy (DP-GAN), and generating synthetic data for downstream classifier augmentation.

---

## 🧪 Specialization Curriculum & Module Matrix

This Specialization is organized into three core courses, taking you from foundational GAN architectures to advanced evaluation and real-world deployment:

| Course / Directory | Operational Focus & Key Technical Deliverables | Key Implementations & Architectures |
| :--- | :--- | :--- |
| **[01 Build Basic GANs](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29)** | **Foundations & Core Architectures**: Adversarial training dynamics, Deep Convolutional GANs (DCGAN), Wasserstein loss with Gradient Penalty (WGAN-GP), and Conditional/Controllable GANs. | • BCE Minimax Loss<br>• DCGAN Transposed Convolutions<br>• WGAN-GP Lipschitz Constraint<br>• One-Hot Class Conditioning |
| **[02 Build Better GANs](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29)** | **Evaluation, Bias & StyleGAN**: Quantitative metrics (FID, IS, Precision/Recall, PPL), VAEs & the Generative Trilemma, ML bias auditing, NeRF, and StyleGAN1–3 building blocks. | • Inception-v3 Deep Feature Extraction<br>• Matrix $\text{sqrtm}$ FID Calculation<br>• VAE Reparameterization Trick<br>• StyleGAN $z \to w$ Mapping & AdaIN |
| **[03 Apply GANs](./03%20Apply%20Generative%20Adversarial%20Networks%20%28GANs%29)** | **Applications, Translation & Privacy**: Generative data augmentation, face de-identification, DP-GANs, paired Pix2Pix translation, and unpaired CycleGAN style transfer. | • Downstream Classifier Augmentation<br>• U-Net Skip Connections & PatchGAN<br>• Dual-Loop Cycle Consistency Loss<br>• GauGAN (SPADE) & Pix2PixHD |

---

## 💡 Visual Pipeline Reference

The unified mathematical progression and architectural lineage across the 3-course series:

- **Adversarial Foundation to Style Disentanglement Evolution**:

  $$\begin{aligned}
  \text{Basic GAN (Course 1): } &z \sim \mathcal{N}(0, I) \longrightarrow \mathbf{G}_{\text{Deconv}} \longrightarrow \hat{x} \longrightarrow \mathbf{D}_{\text{Conv}} \longrightarrow \text{Sigmoid Scalar Logit} \\
  \text{StyleGAN (Course 2): } &z \xrightarrow{\mathbf{\text{MLP}}_{\times 8}} w \in \mathbb{W} \longrightarrow \text{AdaIN / ModDemod Modulation at Resolution Blocks} \longrightarrow \text{Disentangled Visuals} \\
  \text{CycleGAN (Course 3): } &x \in X \xrightarrow{\mathbf{G}_{X \to Y}} \hat{y} \xrightarrow{\mathbf{G}_{Y \to X}} \hat{x}_{\text{recon}} \implies \mathcal{L}_{\text{cyc}} = \|x - \hat{x}_{\text{recon}}\|_1
  \end{aligned}$$

- **Generative Paradigms & Evaluation Benchmark**:

| Architecture | Core Metric / Loss | Key Advantage | Primary Limitation |
| :--- | :--- | :--- | :--- |
| **DCGAN / WGAN-GP** | Wasserstein Distance + $\lambda \|\nabla D\|_2$ | Stable adversarial training; prevents mode collapse | Global latent space entanglements |
| **Variational Autoencoder** | $\text{MSE / BCE} + D_{\text{KL}}(q_\phi(z|x) \parallel p(z))$ | Explicit log-likelihood ELBO bound; stable training | Blurry reconstruction outputs |
| **StyleGAN1–3** | Fréchet Inception Distance (FID) | Unmatched image fidelity & latent attribute control | High computational training requirements |
| **Pix2Pix** | Conditional GAN + $\lambda \|Y - \hat{Y}\|_1$ | Preserves multi-scale spatial structure | Requires strictly aligned training pairs |
| **CycleGAN** | Adversarial Loss + $\lambda_{\text{cyc}} \mathcal{L}_{\text{cyc}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}}$ | Unpaired domain-to-domain style translation | Cannot make drastic structural geometry changes |

---

## 🎯 Technical Skills Architecture

### 📊 Generative Theory & Applied Mathematics
- **Game-Theoretic Optimization:** Formulating adversarial equilibrium dynamics, analyzing gradient vanishing in min-max objectives, and implementing Wasserstein distance constraints.

- **Statistical Distribution Comparison:** Deriving matrix algebra calculations for Fréchet distance over Gaussian feature embeddings and measuring Kullback-Leibler (KL) divergence.

- **Disentangled Geometry & Fairness:** Linearizing curved latent manifolds ($w$-space) for continuous feature editing and applying statistical demographic parity metrics to evaluate dataset bias.


### 🤖 Applied PyTorch & High-Performance Engineering
- **Custom PyTorch Deep Networks:** Engineering end-to-end custom models including DCGAN, WGAN-GP, Conditional GANs, VAEs, StyleGAN, U-Net, PatchGAN, and ResNet generators.

- **Complex Training Workflows:** Managing multi-loss backward passes, dynamic gradient penalty hooks, intermediate activation feature hooks, and multi-network state tracking ($4$ models in CycleGAN).

- **Production Computer Vision Pipelines:** Preprocessing image datasets, implementing synchronized spatial transformations, managing historical image replay buffers, and benchmarking downstream classification gains.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Computer Vision & Transforms | Metrics & Scientific Computing | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Transforms_&_Models-0056D2?style=flat&logo=python&logoColor=white) | ![SciPy](https://img.shields.io/badge/SciPy_&_NumPy-Matrix_Algebra-8CAAE6?style=flat&logo=scipy&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
