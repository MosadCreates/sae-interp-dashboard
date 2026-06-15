# Sparse Autoencoders for Mechanistic Interpretability — Theory

This document explains the mathematical and conceptual foundations of Sparse Autoencoders (SAEs) as applied to transformer interpretability. It is written for readers with a background in deep learning but no prior exposure to mechanistic interpretability.

---

## 1. The Superposition Hypothesis

### 1.1 The Problem: Neurons Are Not Features

A naive view of neural networks says: *each neuron detects one concept*. If this were true, we could understand a model by simply looking at which inputs activate each neuron most strongly. Early interpretability work (e.g., for vision models) took this approach.

But this view breaks for language models. Consider a single neuron in GPT-2's MLP layer that activates for all of the following:

| Input context | Activation |
|---|---|
| "The Dallas **Cowboys** won the game" | 12.4 |
| "I live in **Texas** near Austin" | 8.7 |
| "She listens to country **music** every day" | 6.2 |
| "The **rodeo** starts next week" | 5.1 |

This neuron (sometimes called the *"Dallas Cowboys neuron"*) fires for football, US states, country music genres, and Western cultural events. It is **polysemantic** — it responds to multiple unrelated concepts.

### 1.2 Anthropic's Superposition Hypothesis

Elman (1990) observed that neural networks represent more features than they have dimensions by **compressing** overlapping features into the same neurons. Anthropic formalised this as the **superposition hypothesis** (Elhage et al., 2022):

> Neural networks represent features as **directions in activation space**, not as individual neurons. When the model needs to represent more features than it has dimensions, these feature-directions become **superposed** — they overlap in ways that interfere with each other.

Concretely, in a $d$-dimensional residual stream, the model can represent:
- **Up to $d$ orthogonal features** without interference
- **Up to $\mathcal{O}(d^2)$ features with interference** if features are sparse (most features are inactive for most inputs)

The key insight: **sparsity is what makes superposition possible**. If only $k \ll d$ features are active at any given time, they can share dimensions without colliding, much like how multiple conversations can coexist in a room if only a few people speak at once.

### 1.3 Implications for Interpretability

Superposition means that **direct neuron analysis is fundamentally misleading**:
- A single neuron may participate in representing dozens of features
- A single feature may be distributed across many neurons
- Neuron-level analysis will both miss features and conflate unrelated ones

This is the problem that Sparse Autoencoders solve.

---

## 2. What Is a Sparse Autoencoder?

### 2.1 Conceptual Framework

An SAE is a **learned dictionary** that transforms the model's activations from the standard basis (neurons) to a **sparse, overcomplete basis** (features).

Given an input activation vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ (e.g., the residual stream at some layer, shape `[768]` for GPT-2 Small), the SAE learns:
- **Encoder**: maps $\mathbf{x}$ to a sparse hidden code $\mathbf{z} \in \mathbb{R}^{d_{\text{sae}}}$ where $d_{\text{sae}} \gg d_{\text{model}}$ (overcomplete)
- **Decoder**: reconstructs $\mathbf{x}$ from $\mathbf{z}$ as a linear combination of dictionary vectors
- **Sparsity penalty**: encourages each $\mathbf{z}$ to have few non-zero entries

The decoder columns $\{\mathbf{d}_j\}_{j=1}^{d_{\text{sae}}}$ with $\mathbf{d}_j \in \mathbb{R}^{d_{\text{model}}}$ are the **dictionary vectors** — each represents one feature direction in activation space. The hidden code $\mathbf{z}$ tells us which features are present in the current activation.

### 2.2 Forward Pass

Let:
- $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ — input activation (residual stream at a given layer)
- $\mathbf{W}_{\text{enc}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{sae}}}$ — encoder weights
- $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{d_{\text{sae}} \times d_{\text{model}}}$ — decoder weights
- $\mathbf{b}_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}}}$ — encoder bias
- $\mathbf{b}_{\text{pre}} \in \mathbb{R}^{d_{\text{model}}}$ — pre-encoder bias (centering)

The SAE forward pass is:

$$
\begin{aligned}
\mathbf{x}_{\text{centered}} &= \mathbf{x} - \mathbf{b}_{\text{pre}} \\
\mathbf{z} &= \text{ReLU}(\mathbf{x}_{\text{centered}}^\top \mathbf{W}_{\text{enc}} + \mathbf{b}_{\text{enc}}) \\
\hat{\mathbf{x}} &= \mathbf{W}_{\text{dec}}^\top \mathbf{z} + \mathbf{b}_{\text{pre}}
\end{aligned}
$$

Where:
- **$\mathbf{b}_{\text{pre}}$** centres the input distribution (subtracts the mean activation). This is critical because it lets the SAE focus on **deviations from the mean** rather than wasting dictionary elements on representing the mean itself.
- **ReLU non-linearity** ensures non-negative feature activations. This is standard in SAE literature (Anthropic, 2023; Bricken et al., 2023) — non-negative activations correspond to "feature is present" vs "feature is absent", with no negative firing.
- **Untied weights**: $\mathbf{W}_{\text{enc}}$ and $\mathbf{W}_{\text{dec}}$ are **not** transposes of each other (unlike tied autoencoders). The encoder learns feature **detectors** while the decoder learns the feature **directions** — these are related but distinct roles.

### 2.3 Loss Function

The SAE is trained to minimise:

$$
\mathcal{L} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{reconstruction loss}} + \lambda \underbrace{\|\mathbf{z}\|_1}_{\text{L1 sparsity penalty}}
$$

#### Reconstruction Loss $\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$

This is the mean-squared error between the original activation and the SAE's reconstruction. It ensures that the compressed representation $\mathbf{z}$ retains enough information to faithfully reproduce $\mathbf{x}$. Without this term, the SAE would trivially set $\mathbf{z} = \mathbf{0}$.

#### L1 Sparsity Penalty $\lambda \|\mathbf{z}\|_1$

This penalises the sum of absolute feature activations, encouraging $\mathbf{z}$ to be sparse (most entries zero). The hyperparameter $\lambda$ controls the **sparsity-fidelity tradeoff**.

### 2.4 Evaluation Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **L0 sparsity** | $\frac{1}{N}\sum_{i=1}^N \|\mathbf{z}^{(i)}\|_0$ | Mean number of active features per token (true sparsity) |
| **L1 loss** | $\frac{1}{N}\sum_{i=1}^N \|\mathbf{z}^{(i)}\|_1$ | Proxy for sparsity (L0 is non-differentiable) |
| **Explained variance** | $1 - \frac{\text{Var}(\mathbf{x} - \hat{\mathbf{x}})}{\text{Var}(\mathbf{x})}$ | Fraction of activation variance recovered by the SAE |
| **Feature frequency** | $\frac{1}{N}\sum_{i=1}^N \mathbb{1}[z_j^{(i)} > 0]$ | How often feature $j$ fires |

**Important**: L0 norm is the true sparsity metric we care about, but it is non-differentiable (it counts non-zero entries). L1 norm is a convex relaxation that serves as a differentiable proxy. A well-trained SAE achieves both low L0 and high explained variance.

---

## 3. Why L1 Specifically?

### 3.1 The Geometry of L1 vs L2

The choice of L1 (not L2) for the sparsity penalty is not arbitrary — it follows from the geometry of the unit balls in different norms.

Consider a 2D space where we want a sparse representation (one entry zero):

| Penalty | Constraint set | Geometry | Induces sparsity? |
|---|---|---|---|
| L2 | $\|\mathbf{z}\|_2 \leq c$ | Circle | No |
| L1 | $\|\mathbf{z}\|_1 \leq c$ | Diamond | Yes |

**Why L2 does not produce sparsity**: The L2 ball is rotationally symmetric. If you minimise reconstruction error subject to $\|\mathbf{z}\|_2 \leq c$, the optimal solution typically spreads weight across all coordinates — the gradient direction points towards distributing error evenly.

**Why L1 produces sparsity**: The L1 ball has "corners" on the coordinate axes. When the reconstruction gradient points toward a point outside the L1 ball, the optimal constrained solution lies at a corner — i.e., on a coordinate axis where some entries are exactly zero. This is the mechanism by which L1 induces **exact zeros** in the feature vector.

Concretely, for a 2D least-squares problem:
- L2 regularisation: $\min \|A\mathbf{z} - \mathbf{b}\|_2^2 + \lambda \|\mathbf{z}\|_2^2$ → dense $\mathbf{z}$ with small entries
- L1 regularisation: $\min \|A\mathbf{z} - \mathbf{b}\|_2^2 + \lambda \|\mathbf{z}\|_1$ → sparse $\mathbf{z}$ with exact zeros

This is the same reason LASSO regression uses L1 while Ridge regression uses L2.

### 3.2 The Sparsity-Fidelity Tradeoff

The hyperparameter $\lambda > 0$ controls the balance:

| $\lambda$ | Regime | L0 sparsity | Explained variance | Behaviour |
|---|---|---|---|---|
| $\lambda = 0$ | No sparsity | $d_{\text{sae}}$ (dense) | Highest (≈ PCA) | Every feature fires on every input — no interpretability |
| $\lambda$ optimal | Balanced | $k \ll d_{\text{sae}}$ | High (≈ 90-95%) | Monosemantic features emerge |
| $\lambda \to \infty$ | Over-regularised | 0 (all zero) | Lowest (≈ 0%) | All features dead — trivial solution |

Finding the right $\lambda$ is the central hyperparameter optimisation problem in SAE training. Too low → features are dense and polysemantic. Too high → features die out completely (the "dead features" problem, see §4).

### 3.3 Connection to Sparse Coding (Olshausen & Field, 1996)

The SAE formulation is a direct descendant of **sparse coding** from computational neuroscience. Olshausen and Field showed that the mammalian primary visual cortex (V1) learns Gabor-filter-like receptive fields when trained with a sparsity constraint on natural images. The SAE applies the same principle to transformer activations: **features that are sparse in the data distribution are the features the model actually uses internally**.

---

## 4. The Dead Features Problem

### 4.1 What Are Dead Features?

A feature $j$ is **dead** if its encoder weight row $\mathbf{W}_{\text{enc}}[:, j]$ never wins the competition for any input in the training corpus. That is:

$$
z_j = \text{ReLU}(\mathbf{x}_{\text{centered}}^\top \mathbf{W}_{\text{enc}}[:, j] + b_{\text{enc}, j}) = 0 \quad \forall \mathbf{x} \in \mathcal{D}
$$

Once dead, a feature receives zero gradient (ReLU's gradient is zero for negative inputs), and it **stays dead forever** under standard training.

### 4.2 Why Dead Features Matter

Dead features waste representational capacity. If 50% of your $d_{\text{sae}} = 6144$ features are dead, you are effectively training a model with only ~3000 features — you paid for 6144 but are using half. Dead feature rates above 30-50% are common in naively trained SAEs.

### 4.3 Neuron Resampling (Anthropic, 2023)

The standard solution is **neuron resampling**:

1. **Track alive features**: maintain a buffer of which features have fired (activation > 0) in the last $N$ training steps (typically $N = 12,500$).
2. **Identify dead features**: every $M$ steps (typically $M = 25,000$), find features that have not fired in the buffer.
3. **Resample dead features**:
   - Randomly sample high-loss training examples (inputs the SAE currently reconstructs worst)
   - Set the dead encoder direction to the normalised input activation: $\mathbf{W}_{\text{enc}}[:, j] = \frac{\mathbf{x}_{\text{high-loss}}}{\|\mathbf{x}_{\text{high-loss}}\|}$
   - Set the corresponding decoder column to the same direction: $\mathbf{W}_{\text{dec}}[j, :] = \mathbf{W}_{\text{enc}}[:, j]^\top$
   - Reset the encoder bias: $b_{\text{enc}, j} = 0$

This gives dead features a "second chance" by re-initialising them to directions that are actually present in the training data.

---

## 5. The Decoder Normalisation Constraint

### 5.1 The Problem: Vanishing Features

Without constraints, the SAE can find a trivial solution to minimise L1 loss: make the feature activations $\mathbf{z}$ small by making the decoder columns $\mathbf{W}_{\text{dec}}[j, :]$ large. Since:

$$
\hat{\mathbf{x}} = \sum_{j=1}^{d_{\text{sae}}} z_j \cdot \mathbf{W}_{\text{dec}}[j, :] + \mathbf{b}_{\text{pre}}
$$

Multiplying $z_j$ by $\alpha < 1$ and $\mathbf{W}_{\text{dec}}[j, :]$ by $1/\alpha$ leaves the reconstruction unchanged but reduces $\|\mathbf{z}\|_1$ by a factor of $\alpha$. The L1 penalty would drive $\mathbf{z} \to \mathbf{0}$ and $\|\mathbf{W}_{\text{dec}}[j, :]\| \to \infty$ — a degenerate solution.

### 5.2 The Solution: Unit-Norm Decoder Columns

The fix is to constrain decoder columns to unit Euclidean norm after every gradient step:

$$
\mathbf{W}_{\text{dec}}[j, :] \leftarrow \frac{\mathbf{W}_{\text{dec}}[j, :]}{\|\mathbf{W}_{\text{dec}}[j, :]\|_2} \quad \forall j
$$

This prevents the L1 loss from shrinking activations by growing decoder norms. The normalisation is applied **after every gradient step** (not just at initialisation), enforced by a `normalize_decoders()` call in the training loop.

### 5.3 Why Not Weight Decay?

Weight decay (L2 regularisation on parameters) is incompatible with the decoder normalisation constraint. Weight decay would penalise the decoder weights, but since they are re-normalised to unit norm every step, weight decay would create a tug-of-war: the optimiser grows weights to counteract weight decay, then normalisation resets them. This wastes gradient steps and degrades training. Standard practice is to use **Adam without weight decay** for SAE training.

---

## 6. Why Middle-to-Late Layers?

The choice of which GPT-2 layer to extract activations from is important. We target **layer 8 of 12** (roughly the 2/3 point) by default for several reasons:

| Layer range | Characteristics | Suitability for SAE |
|---|---|---|
| **Early (1-4)** | Token-level, syntactic features. Low abstraction. | Poor — features are too local and numerous |
| **Middle (5-8)** | Mix of syntax and semantics. Higher-level concepts emerge. | **Best** — rich feature variety |
| **Late (9-12)** | Highly task-specific, next-token-prediction features | Moderate — features are task-focused |

Middle layers strike the best balance: they contain semantically meaningful features (names, dates, concepts) without being dominated by the next-token prediction objective that dominates the final layers.

---

## 7. Worked Example: Forward Pass Shapes

For GPT-2 Small feeding through a SAE with expansion factor 8:

| Tensor | Shape | Description |
|---|---|---|
| $\mathbf{x}$ | `[768]` | Residual stream activation at layer 8, position $t$ |
| $\mathbf{x}_{\text{centered}}$ | `[768]` | Mean-centred activation |
| $\mathbf{W}_{\text{enc}}$ | `[768, 6144]` | Encoder weight matrix |
| $\mathbf{b}_{\text{enc}}$ | `[6144]` | Encoder bias |
| $\mathbf{z}$ | `[6144]` | Sparse feature activations (most entries = 0) |
| $\mathbf{W}_{\text{dec}}$ | `[6144, 768]` | Decoder weight matrix (columns = feature directions) |
| $\hat{\mathbf{x}}$ | `[768]` | Reconstructed activation |

With a batch of 4096 token positions, $\mathbf{x}$ has shape `[4096, 768]` and $\mathbf{z}$ has shape `[4096, 6144]`. L0 sparsity would be, for example, `~50` — meaning out of 6144 possible features, only about 50 are active for any given token.

---

## References

- Olshausen, B. A., & Field, D. J. (1996). "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature*.
- Elhage, N., et al. (2022). "Toy Models of Superposition." *Transformer Circuits Thread*.
- Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." *Transformer Circuits Thread*.
- Elman, J. L. (1990). "Finding structure in time." *Cognitive Science*.
- Cunningham, H., et al. (2023). "Sparse Autoencoders for Highly Scalable Feature Discovery in Language Models." — *the first paper to apply SAEs to LLMs at scale*.
- Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Anthropic*.
