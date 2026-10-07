
# 🦓 Week 3: Unpaired Image-to-Image Translation with CycleGAN

Welcome to Week 3 of **Apply Generative Adversarial Networks (GANs)** (Course 3 of the **DeepLearning.AI GANs Specialization**)! This module addresses the core limitation of paired image translation: the scarcity and high cost of aligned training pairs. You will explore unpaired image-to-image translation pioneered by Zhu et al., where models learn mapping functions between two distinct domains $X$ and $Y$ without direct one-to-one image correspondences. You will construct the complete dual-generator CycleGAN system in PyTorch—mastering Cycle Consistency Loss, Identity Loss, residual generator blocks, and implementing a bi-directional translation pipeline between horses and zebras.

---

## 📝 Core Technical Objectives
- **Unpaired Image-to-Image Translation Paradigm:** Translating visual styles between domain $X$ (e.g., horses) and domain $Y$ (e.g., zebras) using two unaligned image sets $\{x_i\}_{i=1}^N \subset X$ and $\{y_j\}_{j=1}^M \subset Y$. Formulating two forward/backward mappings $G_{X \to Y}: X \to Y$ and $G_{Y \to X}: Y \to X$ alongside domain-specific discriminators $D_Y$ and $D_X$.

- **Cycle Consistency Loss Mechanics:** Preventing mode collapse and preserving underlying spatial structure (pose, background geometry) without paired target images. Enforcing that passing an image through both translation generators returns it to its original form ($x \to G_{X \to Y}(x) \to G_{Y \to X}(G_{X \to Y}(x)) \approx x$):

  $$\mathcal{L}_{\text{cyc}}(G_{X \to Y}, G_{Y \to X}) = \mathbb{E}_{x \sim p_{\text{data}}(x)}\left[ \Vert{}G_{Y \to X}(G_{X \to Y}(x)) - x\Vert{}_1 \right] + \mathbb{E}_{y \sim p_{\text{data}}(y)}\left[ \Vert{}G_{X \to Y}(G_{Y \to X}(y)) - y\Vert{}_1 \right]$$

- **Identity Loss & Color Preservation:** Regularizing generator mappings by feeding target-domain samples into generators ($G_{X \to Y}(y) \approx y$). Preventing unwarranted global color shifts or unnecessary scene modifications by penalizing deviations from identity mappings:

  $$\mathcal{L}_{\text{identity}}(G_{X \to Y}, G_{Y \to X}) = \mathbb{E}_{y \sim p_{\text{data}}(y)}\left[ \Vert{}G_{X \to Y}(y) - y\Vert{}_1 \right] + \mathbb{E}_{x \sim p_{\text{data}}(x)}\left[ \Vert{}G_{Y \to X}(x) - x\Vert{}_1 \right]$$

- **ResNet-Based Generator Architecture:** Utilizing downsampling convolutional layers, a bottleneck of $6$ or $9$ Residual Blocks (`ResNetBlock`) with instance normalization and skip connections, and upsampling transposed convolutions to process complex style transfers without spatial resolution bottlenecks.

- **Multi-Modal Unpaired Translation (MUNIT):** Decomposing domain representations into disentangled content codes (shared structural space) and style codes (domain-specific appearance vectors) to generate diverse multi-modal outputs from a single input image.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, PyTorch CycleGAN assignment, and specialized optional labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./03%20Apply%20Generative%20Adversarial%20Networks%20%28GANs%29/03_Week/Lecture%20Files/Course-3-Week-3.pdf)** | Formal visual reference covering paired vs. unpaired dataset geometry, dual-generator cyclic translation loops ($X \to Y \to X$), Cycle Consistency distance proofs, Identity Loss regularization, and MUNIT style/content disentanglement. |
| **[CycleGAN Assignment](./03_Apply_Generative_Adversarial_Networks_%28GANs%29/03_Week/Assingment/C3W3_Assignment_CycleGAN.ipynb)** | End-to-end PyTorch implementation of CycleGAN: building ResNet generator blocks, constructing PatchGAN discriminators, composing the 4-model architecture ($G_{AB}, G_{BA}, D_A, D_B$), and executing bi-directional translation on Horse $\leftrightarrow$ Zebra datasets. |
| **[MUNIT Optional Lab](./03_Apply_Generative_Adversarial_Networks_%28GANs%29/03_Week/Assingment/Course-3-Week-3-Optional-Labs-1-MUNIT.ipynb)** | Implementing Multi-Modal Unpaired Image-to-Image Translation (MUNIT): recombining content codes with distinct domain style codes to produce varied visual outputs. |

---

## 💡 Visual Pipeline Reference

The complete dual-generator CycleGAN execution flow and composite loss structure:

- **CycleGAN Dual-Loop Architecture & Loss Composition**:

  $$\begin{aligned}   \text{Domain } X \longrightarrow \mathbf{G_{X \to Y}} \longrightarrow \hat{Y} \longrightarrow \mathbf{G_{Y \to X}} \longrightarrow \hat{X}_{\text{reconstructed}} &\implies \mathcal{L}_{\text{cyc\_X}} = \Vert{}X - \hat{X}_{\text{reconstructed}}\Vert{}_1 \\   \text{Domain } Y \longrightarrow \mathbf{G_{Y \to X}} \longrightarrow \hat{X} \longrightarrow \mathbf{G_{X \to Y}} \longrightarrow \hat{Y}_{\text{reconstructed}} &\implies \mathcal{L}_{\text{cyc\_Y}} = \Vert{}Y - \hat{Y}_{\text{reconstructed}}\Vert{}_1 \\   &\Downarrow \\   \mathcal{L}_{\text{CycleGAN}} = \mathcal{L}_{\text{GAN}}(G_{X \to Y}, D_Y) + \mathcal{L}_{\text{GAN}}(G_{Y \to X}, D_X) &+ \lambda_{\text{cyc}} \mathcal{L}_{\text{cyc}} + \lambda_{\text{id}} \mathcal{L}_{\text{identity}}   \end{aligned}$$

- **Paired (Pix2Pix) vs. Unpaired (CycleGAN) Translation Frameworks**:

| Dimension | Pix2Pix (Paired) | CycleGAN (Unpaired) |
| :--- | :--- | :--- |
| **Dataset Requirement** | Aligned pairs $(x_i, y_i)$ | Two independent, unaligned sets $\{x_i\} \subset X, \{y_j\} \subset Y$ |
| **Model Components** | 1 Generator ($G$), 1 Discriminator ($D$) | 2 Generators ($G_{X \to Y}, G_{Y \to X}$), 2 Discriminators ($D_X, D_Y$) |
| **Primary Structural Loss** | Direct $L_1$ Pixel Distance $\Vert{}y - G(x)\Vert{}_1$ | Cycle Consistency Loss $\Vert{}x - G_{Y \to X}(G_{X \to Y}(x))\Vert{}_1$ |
| **Generator Architecture** | U-Net with Skip Connections | ResNet Bottleneck Blocks |
| **Primary Application** | Map routes $\leftrightarrow$ Satellite, Segmentation $\leftrightarrow$ Photo | Horse $\leftrightarrow$ Zebra, Monet Paintings $\leftrightarrow$ Real Photos |

---

## 🎯 Technical Skills Architecture

### 📊 Unpaired Translation Theory & Cyclic Losses
- **Cycle Consistency Constraints:** Mathematical proof demonstrating how cyclic mapping ($G_{Y \to X}(G_{X \to Y}(x)) \approx x$) restricts the set of admissible target distributions, preventing mode collapse without explicit ground-truth image pairs.

- **Identity Loss Optimization:** Formulating identity penalties to stabilize training dynamics and ensure generators preserve input color distributions when presented with target domain samples.

- **Content vs. Style Disentanglement (MUNIT):** Deconstructing image features into domain-invariant content codes and domain-specific style vectors for non-deterministic translation.


### 🤖 Applied PyTorch & Multi-Model Systems Engineering
- **Dual-Generator / Dual-Discriminator Management:** Constructing and orchestrating training steps across 4 distinct PyTorch networks ($G_{AB}, G_{BA}, D_A, D_B$) with separate optimizer state tracking (`torch.optim.Adam`).

- **ResNet Bottleneck Block Construction:** Writing custom residual layers (`ResNetBlock`) using Reflection Padding, 2D Convolution, Instance Normalization (`nn.InstanceNorm2d`), and element-wise residual addition ($x + f(x)$).

- **Image Replay Buffer Management:** Implementing historical sample buffers (`ImageBuffer`) to store previously generated fake images for discriminator updates, stabilizing adversarial loss oscillations.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Computer Vision & Tensor Operations | Optimization & Replay Buffers | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![Torchvision](https://img.shields.io/badge/Torchvision-Transforms_&_DataLoaders-0056D2?style=flat&logo=python&logoColor=white) | ![NumPy](https://img.shields.io/badge/NumPy-Replay_Buffers_&_Arrays-013243?style=flat&logo=numpy&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |

