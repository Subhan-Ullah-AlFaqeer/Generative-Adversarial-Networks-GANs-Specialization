# ⚖️ Week 3: Wasserstein GANs with Gradient Penalty (WGAN-GP)

Welcome to Week 3 of **Build Basic Generative Adversarial Networks (GANs)** (Course 1 of the **DeepLearning.AI GANs Specialization**)! This module addresses critical GAN training pathologies—specifically mode collapse and vanishing gradients caused by Standard Binary Cross-Entropy (BCE) loss. You will study Earth Mover's Distance (Wasserstein Distance), mathematical enforcement of 1-Lipschitz continuity, and implement a production-grade Wasserstein GAN with Gradient Penalty (WGAN-GP) alongside Spectral Normalization (SN-GAN) in PyTorch.

---

## 📝 Core Technical Objectives
- **Pathologies of Standard BCE Loss & Mode Collapse:** Analyzing how Minimax BCE loss leads to vanishing gradients when the Discriminator becomes too powerful ($D(x) \to 1, D(G(z)) \to 0$), and how $G$ responds by outputting a single low-variance target mode that consistently fools $D$ (Mode Collapse).

- **Earth Mover's Distance & Wasserstein Loss Formulation:** Replacing Jensen-Shannon / KL divergence with the Earth Mover's Distance to measure the minimum cost of transporting probability mass from generated distribution $P_g$ to real distribution $P_r$. Formulating the dual optimization objective using a Critic network $C$ (replacing the Discriminator):

$$\min_{G} \max_{C \in \mathcal{D}} \underset{x \sim P_r}{\mathbb{E}}[C(x)] - \underset{\hat{x} \sim P_g}{\mathbb{E}}[C(G(z))]$$

- **1-Lipschitz Continuity Enforcement via Gradient Penalty (WGAN-GP):** Enforcing the 1-Lipschitz constraint ($\Vert{}\nabla_{\hat{x}} C(\hat{x})\Vert{}_2 \le 1$) by penalizing deviations of the Critic's gradient norm from $1$ along randomly interpolated points $\hat{x} = \epsilon x + (1 - \epsilon) G(z)$ with penalty weight $\lambda = 10$:

$$\mathcal{L}_{\text{Critic}} = \underset{\hat{x} \sim P_g}{\mathbb{E}}[C(G(z))] - \underset{x \sim P_r}{\mathbb{E}}[C(x)] + \lambda \underset{\hat{x} \sim P_{\text{penalty}}}{\mathbb{E}} \left[ \left( \Vert{}\nabla_{\hat{x}} C(\hat{x})\Vert{}_2 - 1 \right)^2 \right]$$

- **Spectral Normalization (SN-GAN):** Studying alternative Lipschitz enforcement by bounding the matrix 2-norm (largest singular value $\sigma(W)$) of each Critic layer weight matrix $\mathbf{W} \to \frac{\mathbf{W}}{\sigma(\mathbf{W})}$ via power iteration (`nn.utils.spectral_norm`).

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, PyTorch WGAN-GP programming assignments, and optional labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Lecture%20Files/Course-1-Week-3.pdf)** | Formal visual reference covering Mode Collapse dynamics, Earth Mover's Distance intuition, 1-Lipschitz slope constraints, Gradient Penalty calculation geometry, and Spectral Normalization power iteration math. |
| **[WGAN-GP Assignment](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Assingment/C1W3_WGAN_GP.ipynb)** | Complete PyTorch implementation of WGAN-GP: computing random linear image interpolations, deriving autograd vector gradients using `torch.autograd.grad()`, calculating the Gradient Penalty loss term, removing Batch Normalization from the Critic (replacing with InstanceNorm/LayerNorm), and training with multiple Critic steps per Generator step ($n_{\text{critic}} = 5$). |
| **[Spectral Normalization Lab](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Assingment/C1W3_SNGAN.ipynb)** | Implementing Spectral Normalization (`nn.utils.spectral_norm`) across Conv2d layers in the Critic network as an alternative to explicit Gradient Penalty calculation. |
| **[ProteinGAN Application Lab](./01%20Build%20Basic%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Assingment/C1W3_Optional-Labs-2-ProteinGAN.ipynb)** | Exploring specialized domain-specific GAN architectures using Wasserstein distance loss for generating biological sequence structures and synthetic protein representations. |

---

## 💡 Visual Pipeline Reference

The 1-Lipschitz Gradient Penalty computational flow alongside the random interpolation sampling geometry:

- **Gradient Penalty Calculation Pipeline**:

$$\begin{aligned}   \begin{cases} x \sim P_r \\ G(z) \sim P_g \end{cases} \longrightarrow \hat{x} = \epsilon x + (1 - \epsilon) G(z) \quad (\text{where } \epsilon \sim U(0, 1)) &\longrightarrow C(\hat{x}) \\   &\xrightarrow{\text{autograd.grad}} \nabla_{\hat{x}} C(\hat{x}) \\   &\longrightarrow \left( \Vert{}\nabla_{\hat{x}} C(\hat{x})\Vert{}_2 - 1 \right)^2 \times \lambda \implies \mathcal{L}_{\text{GP}}   \end{aligned}$$

- **WGAN-GP Training Mechanics vs. Standard GAN**:

| Parameter / Technique | Standard DCGAN | WGAN-GP |
| :--- | :--- | :--- |
| **Output Evaluator Network** | Discriminator $D(x) \in [0, 1]$ (Sigmoid) | Critic $C(x) \in \mathbb{R}$ (Unbounded Logits) |
| **Loss Function** | Binary Cross-Entropy (BCE) | Earth Mover's / Wasserstein Loss |
| **Evaluator Normalization** | Batch Normalization (`BatchNorm2d`) | Instance / Layer Normalization (No Batch dependencies) |
| **Training Steps Ratio** | $1 : 1$ ($D$ updates per $G$ update) | $n_{\text{critic}} = 5 : 1$ ($C$ updates per $G$ update) |
| **Regularization Constraint** | None / Weight Decay | Gradient Penalty ($\lambda = 10$) or Spectral Norm |

---

## 🎯 Technical Skills Architecture

### 📊 Mathematical Optimization & Measure Theory
- **Earth Mover's Continuous Metric:** Proving why Earth Mover's Distance yields smooth continuous gradients everywhere—even when $P_r$ and $P_g$ have disjoint non-overlapping supports where JS/KL divergences saturate to constants.

- **1-Lipschitz Gradient Norm Constraints:** Deriving why a function $f$ is 1-Lipschitz continuous if and only if its derivative norm is bounded by 1 everywhere ($\Vert{}\nabla f(x)\Vert{}_2 \le 1$), and why enforcing $\Vert{}\nabla f(x)\Vert{}_2 = 1$ centered around real/fake interpolations maintains global critic stability.

- **Critic vs. Discriminator Functional Dynamics:** Understanding why the Critic provides meaningful loss metrics that correlate directly with visual sample quality (unlike BCE loss values).


### 🤖 Applied PyTorch & Advanced Autograd Engineering
- **PyTorch Higher-Order Gradient Computation:** Utilizing `torch.autograd.grad(outputs=..., inputs=..., grad_outputs=..., create_graph=True, retain_graph=True)` to take gradients of gradients through intermediate interpolated tensors.

- **Batch Normalization Inter-Sample Correlation Elimination:** Replacing `nn.BatchNorm2d` with `nn.InstanceNorm2d` or `nn.LayerNorm` in the Critic to prevent batch-level sample mixing from violating gradient penalty calculations.

- **Asynchronous Optimization Scheduling:** Structuring asymmetrical training loops where the Critic updates $n_{\text{critic}}$ times for every single Generator step to ensure $C$ stays close to the optimal Wasserstein evaluation function.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Tensor Calculation & Autograd | Visualization & Benchmarking | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Autograd](https://img.shields.io/badge/PyTorch-Autograd_Higher_Order-0056D2?style=flat&logo=python&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/Matplotlib-Loss_&_Progression_Tracking-11557C?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
