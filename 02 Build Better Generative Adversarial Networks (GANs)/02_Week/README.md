# ⚖️ Week 2: GAN Disadvantages & Machine Learning Bias

Welcome to Week 2 of **Build Better Generative Adversarial Networks (GANs)** (Course 2 of the **DeepLearning.AI GANs Specialization**)! This module critically examines the structural trade-offs of GANs against alternative generative modeling paradigms—such as Variational Autoencoders (VAEs), Score-Based/Diffusion Generative Models, and Neural Radiance Fields (NeRF). Additionally, it delves deeply into machine learning fairness, identifying systemic sources of bias in generative datasets, defining mathematical fairness criteria, and auditing pre-trained GAN models to evaluate representation disparities across demographic groups.

---

## 📝 Core Technical Objectives
- **Generative Trilemma & Model Alternatives:** Navigating the generative modeling trade-off between sample quality, mode coverage/diversity, and fast sampling speed:
  - **GANs:** High visual quality, fast sampling speed, but unstable training and mode collapse risks.
  - **VAEs:** Explicit density optimization via the Evidence Lower Bound (ELBO), stable likelihood training, and good mode coverage, but producing blurrier samples due to mean squared error pixel reconstruction losses:

$$\mathcal{L}{\text{ELBO}}(\theta, \phi; x) = \mathbb{E}{q{\phi}(z|x)}[\logp{\theta}(x|z)] - D{\text{KL}}\left(q{\phi}(z|x) \parallel p(z)\right)$$
  - **Score-Based & Diffusion Models:** Estimating data distribution gradients $\nabla_x \log p(x)$ via Langevin dynamics and Stochastic Differential Equations (SDEs), offering exact coverage and supreme fidelity, but suffering from slow iterative sampling steps.

- **Sources of Machine Learning Bias:** Identifying structural bias vectors in dataset generation and model pipelines:
  - **Historical & Representation Bias:** Underrepresentation or stereotyping within source training sets (e.g., celeb datasets like CelebA heavily skewed toward specific skin tones, genders, and age groups).
  - **Measurement & Sampling Bias:** Data collection artifacts, inconsistent labeling criteria, and skewed geographical or sensor distributions.
  - **Algorithmic Bias:** Objective functions exacerbating dominant dataset modes while suppressing minority tail distributions.

- **Mathematical Definitions of Fairness:** Formulating formal fairness metrics for evaluation and debiasing:
  - **Demographic Parity:** Ensuring predictions $\hat{Y}$ are statistically independent of sensitive attributes $A$:
$$P(\hat{Y} = 1 \mid A = 0) = P(\hat{Y} = 1 \mid A = 1)$$
  - **Equalized Odds & Equality of Opportunity:** Ensuring error rates (True Positive Rate / False Positive Rate) remain equal across demographic groups $A$.

- **GAN Debiasing & Feature Auditing:** Measuring conditional generation performance across protected sub-demographics using classifiers to detect class distribution shifts and training re-weighted sampling or adversarial debiasing layers.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, VAE/Bias assignment notebooks, and specialized optional labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Lecture%20Files/Course-2-Week-2.pdf)** | Formal visual reference comparing generative model families (GANs vs. VAEs vs. Score Models), sources of bias, mathematical definitions of fairness, and classification metrics for facial recognition debiasing. |
| **[VAE & Bias Assignment](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Assingment/C2W2_Assignment.ipynb)** | Complete implementation of a Variational Autoencoder (VAE) in PyTorch using the reparameterization trick ($z = \mu + \sigma \odot \epsilon$) and auditing a pre-trained facial generator for classification bias across protected attributes. |
| **[Score-Based Generative Models](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Assingment/C2W2_Optional_Labs_2_Score_Based_Generative_Modeling.ipynb)** | Implementing score matching and Langevin dynamics sampling to navigate score-based density gradients without adversarial training loops. |
| **[GAN Debiasing Lab](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Assingment/C2W2_Optional_Labs_3_GAN_Debiasing.ipynb)** | Implementing dataset re-weighting, dynamic importance sampling, and adversarial debiasing targets to force GANs to sample underrepresented feature combinations equally. |
| **[NeRF: Neural Radiance Fields](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/02_Week/Assingment/C2W2_Optional_Labs_4_Neural_Radiance_Fields_%28NeRF%29.ipynb)** | Synthesizing 3D scenes from 2D views by training a fully connected network to map $(x, y, z, \theta, \phi)$ continuous spatial coordinates to volume density $\sigma$ and view-dependent RGB color. |

---

## 💡 Visual Pipeline Reference

The Variational Autoencoder computation pass featuring the Reparameterization Trick alongside the Generative Trilemma comparison matrix:

- **VAE Reparameterization & ELBO Loss Architecture**:

$$\begin{aligned}
  \text{Input } x \longrightarrow \mathbf{\text{Encoder}}_{\phi}(x) &\longrightarrow \begin{cases} \mu_z \in \mathbb{R}^{d_z} \\ \log \sigma_z^2 \in \mathbb{R}^{d_z} \end{cases} \xrightarrow{\text{Reparameterization: } z = \mu + \sigma \odot \epsilon, \, \epsilon \sim \mathcal{N}(0, I)} z \in \mathbb{R}^{d_z} \\
  z &\longrightarrow \mathbf{\text{Decoder}}_{\theta}(z) \longrightarrow \hat{x} \in \mathbb{R}^{\text{shape}} \\
  &\implies \mathcal{L}_{\text{VAE}} = \text{BCE/MSE}(x, \hat{x}) + D_{\text{KL}}\left(\mathcal{N}(\mu_z, \Sigma_z) \parallel \mathcal{N}(0, I)\right)
  \end{aligned}$$

- **Generative Model Paradigms Comparison Matrix**:

| Generative Architecture | Sampling Speed | Sample Quality | Mode Coverage / Diversity | Key Training Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **GANs** | ⚡ Fast (Single Pass) | ⭐ High (Sharp Details) | ⚠️ Prone to Mode Collapse | Adversarial Minimax Game ($G$ vs. $D$) |
| **VAEs** | ⚡ Fast (Single Pass) | ⚠️ Moderate (Blurry) | ⭐ Excellent (Explicit ELBO) | Encoder/Decoder + $D_{\text{KL}}$ Regularization |
| **Score / Diffusion Models**| 🐌 Slow (Iterative) | ⭐ High (State-of-the-Art)| ⭐ Excellent (Exact Score Mapping) | Denoising Score Matching / SDEs |
| **NeRF (3D Synthesis)** | 🐌 Slow (Ray Rendering)| ⭐ High (Photorealistic) | N/A (3D Scene Reconstruction) | Volume Rendering + Spatial MLP Mapping |

---

## 🎯 Technical Skills Architecture

### 📊 Generative Theory & Fairness Mathematics
- **The Generative Trilemma Mechanics:** Analyzing structural mathematical constraints that force generative models to balance sampling latency, likelihood coverage, and visual output fidelity.

- **Reparameterization Trick Calculus:** Deriving why direct random sampling $z \sim q_{\phi}(z|x)$ prevents backpropagation (stochastic node), and how isolating noise via $z = \mu(\phi) + \sigma(\phi) \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ enables end-to-end autograd calculation $\frac{\partial z}{\partial \phi}$.

- **Statistical Fairness Definitions:** Translating qualitative fairness requirements into precise statistical objectives (Demographic Parity, Equalized Odds) to evaluate machine learning bias.


### 🤖 Applied PyTorch & Model Auditing Engineering
- **Custom VAE Implementation:** Constructing encoder and decoder networks in PyTorch, implementing exact analytical KL divergence for Gaussian distributions ($D_{\text{KL}} = -\frac{1}{2} \sum \left(1 + \log\sigma^2 - \mu^2 - e^{\log\sigma^2}\right)$), and combined reconstruction losses.

- **Demographic Bias Auditing Pipelines:** Writing automated diagnostic code to feed generated synthetic images into pre-trained attribute classifiers, building confusion matrices, and plotting representation skew across subgroup intersections.

- **3D Radiance Field Mapping:** Building NeRF ray-marching routines, positional encoding functions ($\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \dots)$), and volume rendering integration steps in PyTorch.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Generative Architectures | Bias Auditing & Evaluation | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![VAE & NeRF](https://img.shields.io/badge/Generative-VAEs_&_NeRF-0056D2?style=flat&logo=python&logoColor=white) | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Fairness_&_Confusion_Matrices-F7931E?style=flat&logo=scikit-learn&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
