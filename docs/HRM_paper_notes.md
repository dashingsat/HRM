## Notes on “Hierarchical Reasoning Model (HRM)”

Reference: Wang et al., “Hierarchical Reasoning Model,” 2025. [Paper (PDF)](https://arxiv.org/pdf/2506.21734)

### TL;DR
- **What**: HRM is a compact (~27M params) hierarchical recurrent architecture with two coupled modules operating at different timescales: a slow, abstract high-level module (H) and a fast, detailed low-level module (L).
- **How**: Performs latent, multi-step reasoning in a single forward pass without explicit supervision over intermediate steps; no reliance on Chain-of-Thought (CoT) traces.
- **Key mechanism**: “Hierarchical convergence” — the L-module runs multiple rapid steps to a local equilibrium; only then does the H-module advance, after which L is reset and iterates again.
- **Data/efficiency**: Trains effectively with only ~1,000 samples, no pretraining, and achieves strong stability/efficiency despite deep effective computation.
- **Results**: Near-perfect on Sudoku-Extreme and Maze-Hard; outperforms much larger models on ARC-AGI despite no CoT data and far fewer parameters.

### Motivation
- **Limitations of standard Transformers**: Fixed depth constrains algorithmic reasoning capacity and places models in low computational complexity classes; simply scaling width does not address depth-bound tasks.
- **CoT limitations**: Brittle, human-defined step sequences; high token and data costs; error propagation from misordered steps.
- **Latent reasoning**: Shift computation from external token sequences to internal hidden-state dynamics for efficiency and robustness.

### Architecture Overview
- **Two recurrent modules**:
  - **H-module (slow timescale)**: Plans and guides abstract reasoning.
  - **L-module (fast timescale)**: Executes rapid, detailed computations conditioned on H.
- **Coupling and schedule**:
  - L iterates several steps until reaching a local equilibrium.
  - After L stabilizes, H performs a single (slower) update.
  - L is then reset and runs again under the updated H context.
- **Effective computational depth**: Alternating fast L-steps and slow H-steps yields substantial depth in a single forward pass, enabling complex multi-stage reasoning without external CoT.

### Training and Efficiency
- **No pretraining / no CoT supervision**: HRM trains from scratch and learns internal reasoning dynamics without explicit intermediate labels.
- **Low data regime**: Demonstrated strong performance with only ~1k training samples.
- **Stability**: The hierarchical timescale separation and convergence schedule mitigate issues like vanishing gradients that typically hinder very deep or recurrent computation.

### Empirical Results (as reported)
- **Sudoku-Extreme (Full)**: Near-perfect accuracy; increasing depth is necessary, width alone is insufficient for these search/backtracking-heavy puzzles.
- **Maze-Hard**: Finds optimal paths in large mazes.
- **ARC-AGI**: Surpasses state-of-the-art CoT-based approaches and much larger models with longer contexts.
- **General claim**: Executes sequential reasoning “in one forward pass,” solving tasks directly from inputs without generating chains of textual thoughts.

For details and figures, see the paper: [Wang et al., 2025](https://arxiv.org/pdf/2506.21734).

### Concepts and Terms
- **Latent reasoning**: Performing the multi-step computation inside the model’s hidden states rather than via explicit token sequences.
- **Hierarchical convergence**: A control scheme where the fast module settles to a local equilibrium before the slow module advances, improving credit assignment and stability across many iterative steps.

### Comparisons and Positioning
- **Versus CoT**: Removes the dependency on explicit intermediate textual steps; reduces token and data overhead; avoids brittle step-by-step decomposition.
- **Versus standard Transformers**: Addresses depth limitations by embedding many compute steps through coupled recurrences at different timescales.

### Practical Notes / Open Questions
- **Scheduling details**: Exact criteria for L-module “equilibrium,” reset policy, and the number of inner iterations per outer step are key implementation levers.
- **Generalization**: Strong results on symbolic/structured tasks (Sudoku, Maze, ARC); breadth to other domains (e.g., natural language) and scaling behavior remain important future directions.
- **Implementation alignment**: When implementing HRM-like systems, ensure clear separation of timescales, stable inner-loop dynamics, and mechanisms for resetting/conditioning L on H.

### Citation
Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., Abbasi Yadkori, Y. (2025). Hierarchical Reasoning Model. [Paper (PDF)](https://arxiv.org/pdf/2506.21734)


