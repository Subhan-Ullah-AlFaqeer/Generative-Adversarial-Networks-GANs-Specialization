# 🎨 Course 3: Apply Generative Adversarial Networks (GANs)

Welcome to **Course 3** of the **DeepLearning.AI Generative Adversarial Networks (GANs) Specialization**! This final course transitions from theoretical foundations and architectural design into real-world computer vision applications and generative pipelines. You will explore practical implementations across synthetic data augmentation, privacy-preserving face de-identification, paired image-to-image translation, and unpaired style transfer.

Across 3 comprehensive modules, you will build production-grade PyTorch models and operational application pipelines—training conditional GANs to boost downstream classifier performance, constructing U-Net generators and $70 \times 70$ PatchGAN discriminators for Pix2Pix paired image translation, engineering dual-generator CycleGAN networks with Cycle Consistency and Identity losses for unpaired domain translation, and studying advanced variants including GauGAN (SPADE), Pix2PixHD, SRGAN, and MUNIT.

---

## 📝 Core Technical Objectives
- **Generative Augmentation & Privacy Architectures:** Synthesizing rare class samples with conditional GANs to re-balance dataset distributions, evaluating downstream classifier $F_1$-score and decision boundary improvements, and exploring differential privacy guarantees (DP-GAN) and facial de-identification pipelines.

- **Paired Image-to-Image Translation (Pix2Pix):** Mapping source images $X$ directly to target domain $Y$ using paired datasets. Constructing U-Net generators with skip connections to preserve multi-scale spatial geometry, designing local PatchGAN discriminators ($\mathbf{D} \in \mathbb{R}^{B \times 1 \times H' \times W'}$), and optimizing composite losses ($c\text{GAN} + \lambda L_1$):

  $$\mathcal{L}_{\text{Pix2Pix}}(G, D) = \arg\min_G \max_D \mathcal{L}_{c\text{GAN}}(G, D) + \lambda \mathbb{E}_{x, y, z}\left[ \Vert{}y - G(x, z)\Vert{}_1 \right]$$

- **Unpaired Image-to-Image Translation (CycleGAN):** Learning bi-directional style transfers between unaligned image sets $X$ and $Y$ using dual generators ($G_{X \to Y}, G_{Y \to X}$) and dual discriminators ($D_X, D_Y$). Eliminating mode collapse and enforcing structural fidelity using Cycle Consistency and Identity losses:

  $$\mathcal{L}_{\text{cyc}} = \mathbb{E}_{x}\left[ \Vert{}G_{Y \to X}(G_{X \to Y}(x)) - x\Vert{}_1 \right] + \mathbb{E}_{y}\left[ \Vert{}G_{X \to Y}(G_{Y \to X}(y)) - y\Vert{}_1 \right]$$

- **Advanced Generative Synthesizers:** Implementing multi-scale resolution techniques (Pix2PixHD), Spatially-Adaptive Denormalization (GauGAN / SPADE) to prevent feature erasure during semantic map synthesis, perceptual VGG feature losses for Super-Resolution GANs (SRGAN), and disentangled content/style code decomposition (MUNIT).

---

## 🧪 Course Architecture & Module Matrix

This course is structured into 3 sequential modules covering applied data augmentation & privacy, paired image translation, and unpaired style transformation:

| Module / Directory | Operational Focus & Technical Topics | Key Notebooks & Deliverables |
| :--- | :--- | :--- |
| **[01_Week](./01_Week)** | **GANs for Data Augmentation & Privacy**: Downstream model enhancement, conditional data augmentation, privacy preservation, face de-identification, differential privacy (DP-GAN), fingerprinting, and Generative Teaching Networks (GTN). | • `C3W1_Assignment.ipynb`<br>• `C3W1_Optional_Labs_Generative_Teaching_Networks.ipynb` |
| **[02_Week](./02_Week)** | **Image-to-Image Translation with Pix2Pix**: Paired translation frameworks, U-Net generator skip connections, $70 \times 70$ PatchGAN receptive fields, composite $c\text{GAN} + L_1$ loss ($\lambda = 100$), Pix2PixHD, GauGAN (SPADE), and SRGAN. | • `C3W2A_Assignment_Unet.ipynb`<br>• `C3W2B_Assignment_Pix2Pix.ipynb`<br>• `C3W2_Optional_Labs_Pix2PixHD.ipynb`<br>• `C3W2_Optional_Labs_GauGAN.ipynb`<br>• `C3W2_Optional_Labs_Super_resolution GAN)_(SRGAN).ipynb` |
| **[03_Week](./03_Week)** | **Unpaired Translation with CycleGAN**: Dual-generator/dual-discriminator systems ($G_{AB}, G_{BA}, D_A, D_B$), Cycle Consistency Loss, Identity Loss, ResNet residual blocks, image replay buffers, and multi-modal style/content disentanglement (MUNIT). | • `C3W3_Assignment_CycleGAN.ipynb`<br>• `Course-3-Week-3-Optional-Labs-1-MUNIT.ipynb` |

---

## 💡 Visual Pipeline Reference

The operational mathematical formulations and structural workflows governing Course 3:

- **Pix2Pix Paired Translation Architecture Flow**:

  $$\begin{aligned}   \text{Source } X \longrightarrow \mathbf{\text{U-Net Generator}}_G &\longrightarrow \hat{Y}_{\text{fake}} \\   \text{Concat}(X, \hat{Y}_{\text{fake}}) \longrightarrow \mathbf{\text{PatchGAN Discriminator}}_D &\longrightarrow \text{2D Matrix Logits } \mathbb{R}^{B \times 1 \times H' \times W'} \\   &\implies \mathcal{L}_{\text{Total}} = \text{BCE}(\mathbf{D}(X, \hat{Y}_{\text{fake}}), \mathbf{1}) + 100 \cdot \Vert{}Y_{\text{real}} - \hat{Y}_{\text{fake}}\Vert{}_1   \end{aligned}$$

- **CycleGAN Unpaired Dual-Loop & Cycle Consistency Mechanics**:

  $$\begin{aligned}   x \in X \xrightarrow{\mathbf{G_{X \to Y}}} \hat{y} \xrightarrow{\mathbf{G_{Y \to X}}} \hat{x}_{\text{recon}} &\implies \mathcal{L}_{\text{cyc\_X}} = \Vert{}x - \hat{x}_{\text{recon}}\Vert{}_1 \\   y \in Y \xrightarrow{\mathbf{G_{Y \to X}}} \hat{x} \xrightarrow{\mathbf{G_{X \to Y}}} \hat{y}_{\text{recon}} &\implies \mathcal{L}_{\text{cyc\_Y}} = \Vert{}y - \hat{y}_{\text{recon}}\Vert{}_1   \end{aligned}$$

- **Spatially-Adaptive Denormalization (SPADE / GauGAN) Normalization**:

  $$\text{SPADE}(h_{n,c,y,x}, m) = \gamma_{c,y,x}(m) \left( \frac{h_{n,c,y,x} - \mu_c}{\sigma_c} \right) + \beta_{c,y,x}(m)$$

---

## 🎯 Technical Skills Architecture

### 📊 Translation Theory & Generative Applications
- **Structural vs. Unstructured Loss Metrics:** Analyzing why pixel-level $L_1$ penalties combined with PatchGAN adversarial objectives produce sharp paired translations, whereas unpaired domains depend on cyclic topological constraints ($\mathcal{L}_{\text{cyc}}$) to maintain semantic fidelity.

- **Data Augmentation & Downstream Boundary Dynamics:** Evaluating the impact of synthetic augmentation on downstream classification manifolds, identifying when synthetic noise expands generalizeable decision space vs. causing distribution drift.

- **Multi-Modal Style Disentanglement:** Deconstructing image features into domain-invariant content codes and domain-specific style vectors for non-deterministic style synthesis.


### 🤖 Applied PyTorch & Multi-Network Engineering
- **Complex Multi-Model System Engineering:** Constructing and managing training cycles for dual-generator, dual-discriminator systems ($4$ networks in CycleGAN) with dedicated optimizer states and dynamic image replay buffers (`ImageBuffer`).

- **Advanced Convolutional Network Design:** Writing custom PyTorch implementations for U-Net skip-concatenations (`torch.cat`), $70 \times 70$ fully convolutional PatchGAN discriminators, ResNet residual blocks (`ResNetBlock`), and SPADE spatial denormalization layers.

- **End-to-End Image Translation Pipelines:** Building paired and unpaired data loaders with synchronized spatial transformations, multi-loss backward passes, and benchmarking tools for domain translation.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Computer Vision & Image Operations | Optimization & Metrics | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Transforms_&_DataLoaders-0056D2?style=flat&logo=python&logoColor=white) | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F1_Score_&_Metrics-F7931E?style=flat&logo=scikit-learn&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
