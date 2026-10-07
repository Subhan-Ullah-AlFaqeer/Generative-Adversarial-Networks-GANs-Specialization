# 📏 Week 1: Evaluation of GANs

Welcome to Week 1 of **Build Better Generative Adversarial Networks (GANs)** (Course 2 of the **DeepLearning.AI GANs Specialization**)! Evaluating generative models presents a fundamental challenge: pixel-level comparisons (like $L_1$ or $L_2$ norm distances) fail to capture perceptual image quality and semantic variety. This module explores quantitative evaluation metrics designed specifically for GANs. You will master deep feature extraction via pre-trained networks (Inception-v3), study Inception Score (IS), precision/recall for generative distributions, perceptual path length (PPL), and implement Fréchet Inception Distance (FID) in PyTorch to rigorously assess image synthesis fidelity and diversity.

---

## 📝 Core Technical Objectives
- **Inception-v3 Embeddings & Feature Extraction:** Extracting continuous high-level representations by passing raw spatial images through a pre-trained Inception-v3 network. Stripping the final classification layer to leverage the intermediate $2048$-dimensional feature activation space $\mathbf{x} \in \mathbb{R}^{2048}$ for structural perceptual evaluation.

- **Fréchet Inception Distance (FID) Mechanics:** Modeling real ($p_r$) and generated ($p_g$) feature activations as multivariate Gaussian distributions $\mathcal{N}(\mu_r, \Sigma_r)$ and $\mathcal{N}(\mu_g, \Sigma_g)$. Calculating the Wasserstein-2 distance between these distributions in the feature embedding space:

  $$\text{FID} = \Vert{}\mu_r - \mu_g\Vert{}_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

- **Inception Score (IS) & Kullback-Leibler Divergence:** Evaluating quality (sharpness) and diversity (coverage) by analyzing conditional label distribution $p(y \mid x)$ versus marginal label distribution $p(y) = \int p(y \mid x) p(x) dx$ across Inception-v3 predictions:

  $$\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g} \left[ D_{\text{KL}}\left(p(y \mid x) \parallel p(y)\right) \right]\right)$$

- **Precision, Recall & Truncation Trick:** Disentangling fidelity (Precision: how realistic generated samples look relative to real data manifold) from coverage (Recall: how much of the real data distribution $G$ covers). Controlling the fidelity-diversity trade-off during latent sampling using the Truncation Trick on noise vectors $z \sim \mathcal{N}(0, I)$ clipped at threshold $\psi$.

- **Perceptual Path Length (PPL):** Measuring latent space smoothness by evaluating perceptual difference changes along spherical linear interpolations ($\text{slerp}$) of latent vectors $z_1, z_2$, detecting non-linear manifold warpings or abrupt feature jumps.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, PyTorch FID assignment, and perceptual analysis labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/01_Week/Lecture%20Files/Course-2-Week-1.pdf)** | Formal visual reference covering pixel vs. feature space distance comparison, Inception-v3 pooling layer feature extraction, multivariate Gaussian mean/covariance estimations, matrix square root calculations, and Precision/Recall curve geometry. |
| **[Fréchet Inception Distance Assignment](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/01_Week/Assingment/C2W1_Assignment.ipynb)** | Complete PyTorch implementation of FID from scratch: passing real and fake image batches through `torchvision.models.inception_v3`, calculating empirical feature mean vectors $\mu$ and covariance matrices $\Sigma$, computing matrix square roots via SciPy (`scipy.linalg.sqrtm`), and evaluating $G$'s output quality. |
| **[Perceptual Path Length Lab](./02%20Build%20Better%20Generative%20Adversarial%20Networks%20%28GANs%29/01_Week/Assingment/C2W1_Optional-Labs_PPL.ipynb)** | Measuring latent space geometry: calculating perceptual distance differences (using VGG/LPIPS feature networks) along interpolated $z$-trajectories to compute Perceptual Path Length metrics. |

---

## 💡 Visual Pipeline Reference

The feature extraction flow through Inception-v3 alongside the multivariate Gaussian FID calculation pipeline:

- **Inception-v3 Feature Extraction & FID Processing Pipeline**:

$$\begin{aligned}   \text{Real Batch } X_r &\longrightarrow \mathbf{\text{Inception-v3}}_{\text{features}} \longrightarrow \mathbf{F}_r \in \mathbb{R}^{N \times 2048} \longrightarrow \mu_r = \frac{1}{N}\sum \mathbf{F}_r, \quad \Sigma_r = \text{Cov}(\mathbf{F}_r) \\   \text{Fake Batch } X_g &\longrightarrow \mathbf{\text{Inception-v3}}_{\text{features}} \longrightarrow \mathbf{F}_g \in \mathbb{R}^{N \times 2048} \longrightarrow \mu_g = \frac{1}{N}\sum \mathbf{F}_g, \quad \Sigma_g = \text{Cov}(\mathbf{F}_g) \\   &\implies \text{FID} = \Vert{}\mu_r - \mu_g\Vert{}_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)   \end{aligned}$$

- **Quantitative Evaluation Metrics Comparison**:

| Metric | Target Dimension | Key Mathematical Mechanism | Pros / Cons |
| :--- | :--- | :--- | :--- |
| **Inception Score (IS)** | Quality & Diversity | $D_{\text{KL}}\left(p(y \mid x) \parallel p(y)\right)$ via Inception-v3 | ➕ No real dataset needed<br>➖ Insensitive to mode collapse within classes; ignores target distribution $p_r$ |
| **Fréchet Inception Distance (FID)** | Quality & Diversity | Distance between multivariate Gaussians $\mathcal{N}(\mu_r, \Sigma_r)$ and $\mathcal{N}(\mu_g, \Sigma_g)$ | ➕ Highly correlated with human perception; detects mode dropping<br>➖ High sample size requirement ($N \ge 10,000$); assumes normal feature distribution |
| **Precision / Recall** | Separate Quality vs. Coverage | K-NN sphere overlap in feature embedding space | ➕ Disentangles image fidelity from distribution coverage<br>➖ Sensitive to feature extractor choice and hyperparameter $k$ |
| **Perceptual Path Length (PPL)** | Latent Smoothness | $\mathbb{E}\left[\frac{1}{d^2} d_{\text{perceptual}}(G(\text{slerp}(z_1, z_2; t)), G(\text{slerp}(z_1, z_2; t+\epsilon)))\right]$ | ➕ Detects latent space entanglements and sudden manifold jumps<br>➖ Requires specialized perceptual distance networks (LPIPS/VGG) |

---

## 🎯 Technical Skills Architecture

### 📊 Statistical Evaluation Theory & Feature Embeddings
- **Perceptual Embedding Metrics:** Proving why calculating distances in deep convolutional feature spaces captures structural semantics (edges, textures, object compositions) far superior to raw RGB $L_1/L_2$ pixel loss.

- **Multivariate Gaussian Distance Derivatives:** Deriving the Fréchet distance (Wasserstein-2 metric) between two multi-dimensional Gaussian distributions and managing numerical stability during matrix square root computations.

- **KL-Divergence Inception Dynamics:** Explaining how high $D_{\text{KL}}$ in Inception Score requires low entropy for individual predictions $p(y \mid x)$ (high specificity/quality) alongside high entropy for the overall marginal distribution $p(y)$ (high class diversity).


### 🤖 Applied PyTorch & Model Assessment Engineering
- **Feature Layer Hooking & Pre-trained Networks:** Disabling final classification heads in `torchvision.models.inception_v3(pretrained=True)` to extract intermediate $2048$-dimensional feature representations without tracking gradients (`torch.no_grad()`).

- **Matrix Algebra Operations in SciPy/PyTorch:** Computing empirical mean vectors and covariance matrices across tensor batches and invoking `scipy.linalg.sqrtm` for real/complex matrix root extractions.

- **Truncated Normal Sampling Protocols:** Implementing the Truncation Trick (`torch.trunc_normal_` or $z$-scaling) to trade off diversity for higher visual output fidelity during inference.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Evaluation Models & Metrics | Scientific Computing & Math | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Inception_v3-0056D2?style=flat&logo=python&logoColor=white) | ![SciPy](https://img.shields.io/badge/SciPy-Matrix_Algebra_&_sqrtm-8CAAE6?style=flat&logo=scipy&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
