# ReACT
source code for paper "From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging" (AAAI2026)

<p align="center">
  <a href="https://aaai.org/Conferences/AAAI-26/">
    <img src="https://img.shields.io/badge/AAAI-2026-blue.svg" alt="AAAI 2026">
  </a>
</p>

## Paper Information

- **Title**: From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging
- **Abstract**: Model merging combines expert models for multitask performance but faces challenges from parameter interference. This has sparked recent interest in controllable model merging, giving users the ability to explicitly balance performance trade-offs. Existing approaches employ a compile-then-query paradigm, performing a costly offline multi-objective optimization to enable fast, preference-aware model generation. This offline stage typically involves iterative search or dedicated training, with complexity that grows exponentially with the number of tasks. To overcome these limitations, we shift the perspective from parameter-space optimization to a direct correction of the model's final representation. Our approach models this correction as an optimal linear transformation, yielding a closed-form solution that replaces the entire offline optimization process with a single-step, architecture-agnostic computation. This solution directly incorporates user preferences, allowing a Pareto-optimal model to be generated on-the-fly with complexity that scales linearly with the number of tasks. Experimental results show our method generates a superior Pareto front with more precise preference alignment and drastically reduced computational cost.
- **Citation**: If you find this work helpful, please cite our paper:
  ```bibtex
  @inproceedings{
    wujialin2026react,
    title={From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging},
    author={Wu, Jialin and Yang, Jian and Wang, Handing and Wen, Jiajun and Yu, Zhiyong},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
    year={2026}
  }
  ```

## Core Contributions

**Our main contributions are threefold:**
- **Paradigm Shift**: We reframe controllable merging as a representation correction problem, identifying the primary bottleneck as a simple linear distortion rather than a complex parameter conflict.
- **Closed-Form Solution**: We introduce an \textit{on-the-fly} analytical method, deriving what is, to our knowledge, the first \textit{closed-form solution} for controllable model merging that bypasses iterative optimization entirely.
- **Superior Performance & Efficiency**: Extensive experiments show our method achieves a state-of-the-art Pareto front, superior preference alignment, and drastically reduced computational cost, maintaining this strong performance even when using only a fraction of the unlabeled data required by competing methods.

## Core Idea and Methodology

Instead of performing expensive optimization in the parameter space, our method, ReACT, directly corrects the model's final representation. Our core insight is that the discrepancy between the representations from the merged model ($\mathcal{Z}^\text{mtl}_t$) and those from the ideal single-task expert ($\mathcal{Z}^\text{ind}_t$) is primarily a **linear distortion**.

Based on this, we propose a simple linear transformation matrix $W_t$ for each task $t$ to correct this: $\mathcal{\hat{Z}}^\text{mtl}_{t} = W_t \mathcal{Z}^\text{mtl}_{t}$. We find the optimal $W_t$ by minimizing the squared Frobenius norm between the corrected and ideal representations. To improve robustness and prevent overfitting when calibration data is scarce, we regularize the solution towards an optimal orthogonal transformation $W_t^{\text{orth}}$. This structural prior prevents $W_t$ from distorting the geometric structure of the representation space.

Crucially, we extend this single-task correction to the multi-objective setting with user preferences $\mathbf{p}$. By applying Linear Scalarization, we transform the multi-objective problem into a weighted single-objective one. Since each loss function is a convex quadratic function of $W$, this allows us to derive a **unique, analytical, closed-form solution for the Pareto-optimal transformation $W_\mathbf{p}$**. This solution enables instantaneous generation of a preference-aware model. Furthermore, its structure reveals a principled aggregation mechanism: $W_\mathbf{p}$ is a data-dependent weighted average of individual correction maps $\hat{W}_t$. The weighting considers not only user preferences $p_t$ but also a data-dependent matrix $C_t$, which accounts for the feature structure of each task. This means our method naturally gives greater influence to corrections for tasks with more pronounced, high-variance feature structures, unlike naive averaging.

## Key Experimental Results & Highlights

Our experiments are designed to validate ReACT's performance, efficiency, controllability, and the effectiveness of its core design principles.

1.  **Performance and Controllability (Q1)**:
    -   **Table 1: Test accuracies (%) on eight datasets when merging eight ViT-B/32 models.** This table showcases our method's performance against non-controllable baselines and state-of-the-art controllable approaches under three preference scenarios (equal, priority, one-hot).
        -   ReACT provides a substantial uplift over the non-controllable backbones it enhances, such as AMPP (85.4% vs. 81.1%).
        -   It is competitive with Pareto Merging (PM) in balanced settings.
        -   Its advantage grows as preferences become more focused, culminating in the one-hot scenario where ReACT achieves a striking **+5.3% average accuracy gain** over PM (88.9% vs. 83.6%).
        -   **Data Efficiency**: Notably, ReACT using just **10% of the calibration data** still outperforms the fully-resourced PM baseline in the crucial priority and one-hot scenarios.

2.  **Pareto Front and Preference Alignment**:
    -   **Table 2: Performance comparison against Pareto Merging (PM).** This table reports Hypervolume (HV) and Uniformity (U) for both 3-task and 8-task settings.
        -   ReACT achieves substantially higher scores across all metrics. For example, AM+Ours achieves **HV@3 of 83.95** compared to AM+PM's 76.77, and **U@8 of 44.77** compared to AM+PM's 31.98.
        -   This indicates that ReACT not only achieves more precise control but also a more robust and holistic trade-off capability without disproportionately sacrificing non-prioritized tasks.
    -   **Figure 3: Visual evidence for the superior U@3 metrics in Table 2.** This figure visually demonstrates how ReACT (b) exhibits sharp, predictable accuracy peaks at corresponding preference corners, while PM's response (a) is more diffuse and misaligned, explaining ReACT's higher Uniformity.
    -   **Figure 2: Pairwise performance trade-offs within an 8-task merge.** This figure visualizes the Pareto frontiers when varying preferences between pairs of tasks. ReACT (blue) consistently produces superior and more stable Pareto fronts compared to PM (orange), which often fails to generate controllable responses on several critical task pairs (e.g., SUN397-Cars, MNIST-DTD).

3.  **Efficiency and Scalability (Q2)**:
    -   **Figure 4: Scalability comparison.** This plot shows the computation time for the preference-aware stage against the number of tasks.
        -   ReACT demonstrates **linear time complexity**.
        -   For an 8-task merge, ReACT completes in just **0.056 GPU hours**, achieving an approximately 36x speedup over PM and a 208x speedup over MAP.
        -   This exceptional efficiency enables **on-the-fly generation of tailored models** for practical applications.

4.  **Ablation and Mechanism Validation (Q3, Q4)**:
    -   **Figure 5: Accuracy vs. regularization strength $\beta$.** This figure shows that our method is robust to the choice of $\beta$ and highly data-efficient, achieving near-optimal performance with only 10% of test data. Regularization is crucial for preventing overfitting and preserving the representation space's structure.
    -   **Table 3: Comparison of aggregation strategies.** Compared to naive weighted averaging and polar decomposition-based strategies, ReACT's principled, data-aware aggregation achieves a much better overall trade-off, especially for non-priority tasks, empirically validating the importance of the data-dependent weighting matrices $C_t$.
    -   **Figure 6: Linear vs. non-linear correction.** This figure compares our global linear map against the non-linear MLP corrector from Representation Surgery (RS).
        -   While RS appears slightly more robust with minimal data (1%), ReACT's performance rapidly surpasses RS with just 5% of the data and widens its lead thereafter. This demonstrates that our linear approach is more data-efficient in capturing the underlying global distortion once a minimum viable sample set is available.

## 🚀 Code Usage Guide

### Environment Setup
**Coming Soon.**

### Repository Structure
**Coming Soon.**

### Reproducing Experiments
**Coming Soon.**

### Usage Example (Optional)
**Coming Soon.**

## 📥 Datasets and Models

### Datasets
The paper primarily uses the following eight image classification datasets:
-   **SUN397**: Large-scale scene recognition dataset.
-   **Cars (Stanford Cars)**: Fine-grained car classification dataset.
-   **RESISC45**: Remote sensing image scene classification dataset.
-   **EuroSAT**: Satellite image land cover classification dataset.
-   **SVHN (Street View House Numbers)**: Street View House Numbers dataset.
-   **GTSRB (German Traffic Sign Recognition Benchmark)**: German Traffic Sign Recognition Benchmark dataset.
-   **MNIST**: Classic handwritten digits recognition dataset.
-   **DTD (Describable Textures Dataset)**: Describable Textures Dataset.

**Download instructions and preprocessing scripts will be provided soon.**

### Pre-trained Models and Checkpoints
Our work is based on the **CLIP ViT-B/32** visual encoder.
-   **CLIP ViT-B/32**: The pre-trained model can be obtained through the Hugging Face Transformers library or OpenAI's official repository.
-   **Task Expert Models**: The expert models for the eight tasks used in the paper, fine-tuned from CLIP ViT-B/32.
-   **Merged Backbone Models**: Pre-merged backbone models (e.g., obtained via AdaMerging or AMPP methods).

**Download links and loading instructions for all models will be provided soon.**

## Contributors and Contact

### Contributors
-   Jialin Wu (Department of Computer, Rocket Force University of Engineering)
-   Jian Yang (Department of Engineering, Rocket Force University of Engineering)
-   Handing Wang (School of Artificial Intelligence, Xidian University)
-   Jiajun Wen (Department of Computer, Rocket Force University of Engineering)
-   Zhiyong Yu (Department of Computer, Rocket Force University of Engineering) - Corresponding author

### Contact
If you have any questions, suggestions, or find any bugs, please feel free to contact us via:
-   **Email**: wujialin11@nudt.edu.cn (Jialin Wu)
-   **GitHub Issues**: We welcome you to open an issue in this project repository.

## License

This project is open-sourced under the **MIT License**. See the [LICENSE](LICENSE) file for details.
