# Toward a Hybrid Quantum–Classical Transformer with Logarithmically Scaling Attention for Long Sequences

## Abstract

Classical self-attention, a cornerstone of modern transformer architectures, fundamentally suffers from quadratic computational and memory complexity with respect to sequence length. This $O(N^2)$ scaling severely limits its applicability to tasks involving very long sequences, such as entire documents, genomic data, or high-resolution imagery. This paper proposes a hybrid quantum–classical transformer architecture designed to attack this bottleneck at the algorithmic level. The core idea is to replace the standard self-attention mechanism with a custom-designed Parameterized Quantum Circuit (PQC) that implements an attention-like operation whose *quantum* computational cost scales polylogarithmically in the sequence length $N$, leading to an overall attention-head cost of $O\big(N,\mathrm{poly}(\log N)\big)$ under a standard QRAM-like cost model.

We formalize a “Quantum Log-Attention via Conditional Interaction” (QLACI) primitive that uses a query-encoding register, a logarithmic-size index register, and controlled key–value interactions to compute expectation values representing attention-weighted context vectors. These expectation values encode global information over all $N$ positions via a superposition over indices, thereby avoiding explicit $N^2$ pairwise score computation. The QLACI module is integrated into a conventional transformer block, yielding a hybrid quantum–classical transformer that retains classical embeddings, feed-forward layers, and residual/normalization structure.

The contribution of this work is conceptual and architectural rather than empirical. We provide a complexity analysis of the QLACI primitive under explicit assumptions (notably, access to a QRAM-like quantum data structure), outline a concrete hybrid architecture, and specify an evaluation program using quantum simulators and long-sequence benchmarks (text, genomics, time series). We also identify and analyze the main fragilities of the approach—QRAM feasibility and constant factors, barren plateaus and trainability, per-query sequential execution, and measurement overhead—and articulate how these constraints shape any realistic path from asymptotic complexity to wall-clock performance. While classical near-linear approximations to attention (e.g., kernelized or state-space transformers) already achieve strong wall-clock performance on today’s hardware, QLACI is intended as a distinct, theoretically motivated point in the design space: a fully global, non-sparse attention-like primitive with near-linear cost in $N$ in its quantum core, conditional on efficient QRAM-like memory. The aim is to articulate a physically motivated route to near-linear-in-$N$ attention with fully global context, opening a research avenue that connects transformer efficiency, quantum hardware models, and complexity theory.

---

## 1. Introduction

### 1.1. Background: The Rise and Limits of Classical Transformers

The Transformer architecture, introduced by Vaswani et al. in 2017 [CITATION: Vaswani et al., 2017], has reshaped sequence modeling across natural language processing (NLP), computer vision [CITATION: Dosovitskiy et al., 2020], speech processing [CITATION: Gulati et al., 2020], and bioinformatics [CITATION: Rives et al., 2021]. Its core innovation, multi-head self-attention, assigns data-dependent weights to all positions in a sequence when computing each token’s representation, enabling direct modeling of long-range dependencies without recurrence.

The price of this expressivity is the quadratic dependence on sequence length $N$. Computing the attention scores $QK^\top$ for queries $Q \in \mathbb{R}^{N \times d_k}$ and keys $K \in \mathbb{R}^{N \times d_k}$ costs $O(N^2 d_k)$ operations and requires $O(N^2)$ memory to materialize the attention matrix. This produces hard limits in practice:

* **Compute:** Training and inference cost explode for $N \gtrsim 10^4$–$10^5$.
* **Memory:** The $N^2$ attention matrix rapidly saturates accelerator memory.
* **Context:** Practitioners must truncate or segment long inputs, discarding global context that is often semantically crucial (e.g., full legal cases, complete genomes, multi-decade climate time series).

Understanding and relaxing this quadratic bottleneck has become a central theme in transformer research.

### 1.2. The Promise of Quantum Computing for Machine Learning

Quantum computing leverages superposition, entanglement, and interference to realize computation over high-dimensional Hilbert spaces [CITATION: Nielsen & Chuang, 2010]. Quantum Machine Learning (QML) explores how these phenomena can be harnessed for data analysis and pattern recognition [CITATION: Biamonte et al., 2017].

On near-term Noisy Intermediate-Scale Quantum (NISQ) devices, the dominant paradigm is the **Variational Quantum Algorithm** (VQA) [CITATION: Cerezo et al., 2021]. A VQA uses a Parameterized Quantum Circuit (PQC) controlled by classical optimization, forming a hybrid loop: the quantum processor evaluates expectation values of observables, while a classical optimizer updates PQC parameters. PQCs have already been studied for classification, regression, generative modeling, and kernel methods.

Speculative advantages relevant to long-sequence modeling include:

* Access to exponentially large state spaces with $O(\log N)$ qubits.
* Natural representation of global correlations through entanglement.
* The possibility of quantum primitives (e.g., QRAM, quantum inner products) that address or aggregate over $N$ items using $\mathrm{poly}(\log N)$ operations.

These advantages are hardware- and model-dependent and are not automatic; they must be engineered into specific algorithms.

### 1.3. Problem Statement

The quadratic complexity of self-attention has triggered a large body of work on more efficient transformer variants:

* **Sparse attention** [CITATION: Child et al., 2019; Beltagy et al., 2020; Zaheer et al., 2020] reduces the number of key–query pairs, often achieving $O(N \sqrt{N})$ or $O(N \cdot \text{window})$ complexity at the cost of constrained attention patterns.
* **Linear attention** [CITATION: Katharopoulos et al., 2020; Choromanski et al., 2021; Kitaev et al., 2020; Wang et al., 2020] replaces explicit $N^2$ score computation with kernel or hashing tricks, achieving $O(N d)$ complexity but often via approximations that may distort global attention.
* **Recurrent/hierarchical transformers** [CITATION: Dai et al., 2019; Rae et al., 2020] process sequences in segments with memory mechanisms, which preserve long-range information but reduce direct simultaneous access to all positions.

These designs reduce the asymptotic cost but usually impose structural or approximation constraints. A mechanism that (i) preserves truly global interactions and (ii) breaks the $O(N^2)$ barrier in a principled way remains an open target.

### 1.4. Proposed Solution: Hybrid Transformer with Quantum Log-Attention

This work proposes a **hybrid quantum–classical transformer** that replaces the classical self-attention computation with a PQC-based primitive designed for logarithmic dependence on sequence length in its quantum subroutine.

The central mechanism is a **Quantum Log-Attention via Conditional Interaction (QLACI)** circuit. For each query, QLACI:

1. Encodes the query into a quantum state on a fixed number of “query qubits”.
2. Prepares an index register of $\lceil \log_2 N \rceil$ qubits in a coherent superposition over all positions.
3. Uses QRAM-like conditional operations to couple the query register to key–value information addressed by the index register.
4. Produces expectation values that act as attention-weighted context vectors.

Under standard QRAM cost assumptions, this yields a per-query cost that is $\mathrm{poly}(\log N)$ in the quantum part, leading to an overall attention-head scaling of $O\big(N,\mathrm{poly}(\log N)\big)$ with respect to sequence length. The rest of the transformer (embeddings, feed-forward networks, residuals, normalization, output heads) remains classical.

While classical near-linear approximations (Performer-style kernels, state-space models such as S4 and Mamba, and related architectures) already achieve strong wall-clock advantages on existing hardware, they typically do so by baking locality, structured sparsity, or kernel approximations directly into the attention mechanism. QLACI is aimed at a different trade-off: it retains a fully global, non-sparse attention-like operation, and seeks near-linear scaling in $N$ by exploiting coherent superposition over indices—*conditional* on access to efficient QRAM-like memory.

Crucially, this is not a claim of unconditional quantum speedup. The improvement is obtained under explicit architectural and hardware assumptions (notably QRAM) and is contrasted against classical mechanisms that achieve linear or near-linear scaling at the cost of locality or approximations in the attention pattern. Whereas today’s highest-performing long-sequence models (Performer, S4, Mamba, Hyena, etc.) already deliver excellent wall-clock efficiency using approximations or recurrence, QLACI pursues a fundamentally different trade-off: fully exact, non-local, non-recurrent global attention at near-linear cost—contingent on future hardware that supports efficient QRAM-like memory access.

### 1.5. Research Hypothesis

**Hypothesis.** Under a QRAM-like access model, a PQC-based QLACI module can serve as a global attention primitive whose computational cost in sequence length scales as $O\big(N,\mathrm{poly}(\log N)\big)$, while preserving task performance comparable to strong classical baselines on moderate sequence lengths and maintaining robust performance where classical quadratic attention becomes infeasible.

The intent is not to claim a proven asymptotic separation from all classical $O(N)$-time efficient transformers; instead, the claim is that quantum hardware enables a *different point* in the accuracy–complexity–globality trade-off space: fully global interactions at near-linear cost in $N$.

### 1.6. Key Contributions

* **QLACI primitive.** A formally specified PQC-based attention primitive that uses a logarithmic-size index register and conditional interactions to implement a global attention-like operation with $\mathrm{poly}(\log N)$ quantum depth per query under QRAM assumptions.
* **Hybrid architecture.** A complete architectural blueprint for integrating QLACI into transformer blocks, including data flow between classical embeddings, the quantum attention module, and classical feed-forward networks.
* **Complexity analysis.** A detailed complexity analysis with respect to sequence length, including gate counts, qubit requirements, and a clear statement of the assumptions (QRAM, fixed embedding dimensions) under which $O\big(N,\mathrm{poly}(\log N)\big)$ scaling is obtained.
* **Evaluation plan.** A concrete empirical program using quantum simulators and long-sequence benchmarks (long-document NLP, genomics, time series), aimed at validating scaling trends and functional performance against state-of-the-art classical baselines.
* **Fragility analysis.** An explicit analysis of where the proposal is structurally fragile—QRAM feasibility and constant factors, barren plateaus and trainability, per-query sequential execution, and measurement overhead—and how these constraints delimit any realistic path from asymptotics to hardware.
* **Reference implementation and toy experiments.** A small PyTorch + Pennylane prototype instantiating QLACI on a synthetic “needle-in-a-haystack” regression task at $N=16$, together with baseline comparisons and diagnostic plots. These experiments are explicitly small-scale sanity checks and executable specification, *not* evidence of quantum advantage.

### 1.7. Paper Structure

Section 2 reviews related work. Section 3 introduces the QLACI design, complexity analysis, and hybrid transformer architecture, and analyzes practical bottlenecks. Section 4 describes an evaluation program together with the current prototype implementation. Section 5 summarizes expected behaviors, limitations, and future directions.

---

## 2. Related Work

### 2.1. Classical Transformer Architectures and Attention Variants

The original Transformer [CITATION: Vaswani et al., 2017] uses scaled dot-product attention:
[
\text{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
]
with $Q, K, V \in \mathbb{R}^{N \times d}$ derived from token embeddings. The core cost arises from the $QK^\top$ multiplication ($O(N^2 d)$) and storage of the $N \times N$ attention matrix.

#### 2.1.1. Sparse Attention

Sparse variants reduce the number of pairwise interactions:

* **Longformer** [CITATION: Beltagy et al., 2020] uses local sliding windows plus a limited set of global tokens.
* **BigBird** [CITATION: Zaheer et al., 2020] combines local, random, and global attention, achieving theoretical guarantees (e.g., Turing completeness) with block-sparse patterns.
* **Block-sparse transformers** [CITATION: Child et al., 2019] impose structured sparsity on heads.

These designs reduce complexity to $O(N \cdot \text{window})$ or similar, but individual tokens cannot directly attend to all others in one step.

#### 2.1.2. Linear and Approximate Attention

Linear-time variants rewrite attention as:
[
\mathrm{softmax}(QK^\top)V \approx (\phi(Q)\phi(K)^\top)V,
]
where $\phi$ is a feature map:

* **Performer** [CITATION: Choromanski et al., 2021]: positive orthogonal random features.
* **Reformer** [CITATION: Kitaev et al., 2020]: locality-sensitive hashing.
* **Linformer** [CITATION: Wang et al., 2020]: low-rank projections for $K$ and $V$.

These offer $O(N d)$ complexity but at the cost of approximation bias and sometimes degraded modeling of rare, non-local dependencies.

#### 2.1.3. Recurrent and Memory-Augmented Transformers

* **Transformer-XL** [CITATION: Dai et al., 2019] introduces recurrence between segments, extending context beyond the standard window.
* **Compressive Transformer** [CITATION: Rae et al., 2020] compresses old memories.

They trade direct, simultaneous access to all positions for incremental processing with memory.

### 2.2. Quantum Machine Learning and Variational Quantum Algorithms

QML builds learning algorithms on quantum substrates [CITATION: Biamonte et al., 2017].

#### 2.2.1. Data Encoding and Feature Maps

Data encoding is a key design choice:

* **Angle encoding**: map components of $x \in \mathbb{R}^d$ to rotation angles on $d$ qubits [CITATION: Schuld et al., 2018].
* **Amplitude encoding**: encode $x$ into amplitudes of a $\log d$-qubit state, with more complex state preparation [CITATION: Schuld, 2018].
* **Quantum feature maps**: circuits that embed data into high-dimensional Hilbert spaces defining quantum kernels [CITATION: Schuld & Killoran, 2019].

#### 2.2.2. PQCs and Training Pathologies

Parameterized Quantum Circuits (PQCs) consist of sequences of parameterized single- and two-qubit gates [CITATION: Sim et al., 2019]. Challenges include:

* **Expressivity vs. trainability**.
* **Barren plateaus**: exponentially vanishing gradients in deep or unstructured PQCs [CITATION: McClean et al., 2018].
* Differentiation via parameter-shift rules [CITATION: Schuld et al., 2019] or simulator-based autodiff.

#### 2.2.3. Hybrid Quantum–Classical Algorithms

Hybrid algorithms split work: the quantum processor evaluates a PQC; a classical processor optimizes parameters. Applications include:

* Quantum neural networks [CITATION: Farhi & Neven, 2018].
* Quantum generative models [CITATION: Lloyd & Weedbrook, 2018].
* Quantum reinforcement learning [CITATION: Dunjko & Briegel, 2018].

QLACI fits into this paradigm: the quantum part is an attention-like subroutine, while the rest of the model remains classical.

### 2.3. Quantum NLP and Quantum Transformers

Quantum NLP (QNLP) connects compositional semantics and quantum mechanics [CITATION: Zeng et al., 2022].

* Categorical compositional frameworks [CITATION: Coecke et al., 2010].
* Quantum-inspired embeddings [CITATION: Widdows & Perfors, 2006; Meichanetzidis et al., 2020].
* Proposals for quantum attention mechanisms and “quantum transformers” [CITATION: Banchi et al., 2020; Zou et al., 2022].

Existing work often ports classical architectures into quantum form or explores semantic structure, but usually does not tackle the $\Theta(N^2)$ cost of attention head-on via sub-quadratic complexity primitives over classical sequences. The present work targets that scaling bottleneck explicitly, albeit under strong structural assumptions (QRAM).

### 2.4. Complexity Theory and QRAM Assumptions

Quantum algorithms can offer exponential or polynomial speedups for certain problems (Shor [CITATION: Shor, 1994], Grover [CITATION: Grover, 1996]). For data access, **quantum random access memory (QRAM)** [CITATION: Giovannetti et al., 2008] provides coherent access to classically stored data indexed by a quantum superposition of addresses.

Under an idealized QRAM model, reading from an array of size $N$ can be done in $\mathrm{poly}(\log N)$ time per query, with the array stored in classical or quantum hardware. QLACI explicitly assumes such a cost model: the logarithmic dependence in $N$ for the quantum part is conditioned on a QRAM-like primitive. This is a nontrivial hardware assumption and a primary limitation of the proposal.

---

## 3. Methodology

We now define the QLACI primitive, its complexity, the hybrid transformer architecture, and the main practical bottlenecks.

### 3.1. Classical Self-Attention Review

Given $Q, K, V \in \mathbb{R}^{N \times d}$:
[
\text{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
]
The $QK^\top$ multiplication costs $O(N^2 d)$ operations and produces an $N \times N$ matrix. For multi-head attention with $h$ heads, dimensions are reshuffled, but the $N^2$ dependence remains.

The core scaling question is: can we compute an attention-like mapping from $(Q,K,V)$ to $N$ context vectors without ever forming or implicitly evaluating all $N^2$ pairwise scores, under a quantum cost model?

### 3.2. QLACI: Logarithmic PQC-Based Attention

QLACI replaces explicit pairwise score computation with a quantum expectation over a superposition of indices. Figure 1 provides a circuit-level schematic of the primitive.

#### 3.2.1. Data Encoding

We separate encoding for queries and for key–value storage.

* **Query encoding.** For each query vector $Q_i \in \mathbb{R}^{d_q}$, define an encoding circuit
  [
  U_{\mathrm{enc,Q}}(Q_i): |0\rangle^{\otimes q_Q} \mapsto |\psi_Q(Q_i)\rangle,
  ]
  where $q_Q = d_q$ for straightforward angle encoding. The rotation angles are affine functions of $Q_i$.

* **Key–value access.** We assume that classical key–value pairs $(K_j, V_j)$, $j \in {1,\dots,N}$, are stored in a QRAM-like structure so that a quantum address $|j\rangle$ on an index register can trigger controlled operations that depend on $(K_j, V_j)$. This assumption is explicit. We do not attempt to implement QRAM from first principles; we treat it as a primitive with $O(\mathrm{poly}(\log N))$ cost per coherent access.

#### 3.2.2. Circuit Registers

QLACI uses:

* **Query register $R_Q$**: $q_Q$ qubits holding $|\psi_Q(Q_i)\rangle$.
* **Index register $R_{\mathrm{idx}}$**: $q_{\mathrm{idx}} = \lceil \log_2 N \rceil$ qubits.
* **Key–value ancilla $R_{KV}$**: $q_{KV}$ qubits, constant in $N$.
* **Output register $R_{\mathrm{out}}$**: $q_{\mathrm{out}}$ qubits, constant in $N$.

#### 3.2.3. QLACI Circuit Structure

For each query $Q_i$, QLACI executes:

1. **Initialization.**

   * Prepare $R_Q$ via $U_{\mathrm{enc,Q}}(Q_i)$.
   * Initialize $R_{\mathrm{idx}}$ into uniform superposition:
     [
     |0\rangle^{\otimes q_{\mathrm{idx}}} \xrightarrow{H^{\otimes q_{\mathrm{idx}}}} \frac{1}{\sqrt{N}}\sum_{j=0}^{N-1} |j\rangle.
     ]
   * Initialize $R_{KV}$ and $R_{\mathrm{out}}$ to $|0\rangle$.

2. **Conditional key–value interaction.**

   * Apply a parameterized interaction circuit
     [
     U_{\mathrm{interact}}(\theta): R_Q \otimes R_{\mathrm{idx}} \otimes R_{KV} \to R_Q \otimes R_{\mathrm{idx}} \otimes R_{KV},
     ]
     built from:

     * Controlled calls to a QRAM-like primitive that, conditioned on $|j\rangle$ in $R_{\mathrm{idx}}$, applies a feature map $U_F(K_j,V_j)$ on $R_{KV}$.
     * Entangling gates (e.g., CNOT, CZ) between $R_Q$ and $R_{KV}$, parameterized by $\theta$.

   Conceptually, after this step, the global state encodes, in superposition, interactions between the fixed query $Q_i$ and all $(K_j, V_j)$.

3. **Score aggregation.**

   * Apply a parameterized aggregation circuit
     [
     U_{\mathrm{score}}(\phi): R_Q \otimes R_{KV} \otimes R_{\mathrm{out}} \to R_Q \otimes R_{KV} \otimes R_{\mathrm{out}},
     ]
     whose purpose is to map the entangled query–key–value state into observables on $R_{\mathrm{out}}$.

4. **Measurement.**

   * Measure an observable $O$ (typically tensor products of Pauli operators) on $R_{\mathrm{out}}$ to obtain an expectation vector
     [
     c_i = \mathbb{E}[O] \in \mathbb{R}^{d_{\mathrm{ctx}}},
     ]
     where $d_{\mathrm{ctx}}$ is the context dimension. This $c_i$ plays the role of the attention output for query $i$.

The key point is that, because $R_{\mathrm{idx}}$ spans all $j$ simultaneously, $c_i$ aggregates contributions from all positions in a single circuit execution, instead of iterating over $j$ explicitly.

QLACI does **not** compute the full $N \times N$ attention matrix; it directly produces the context vectors ${c_i}_{i=1}^N$.

#### 3.2.4. Complexity with Respect to Sequence Length

We treat embedding dimensions and PQC widths as constants with respect to $N$.

* **Qubits.**

  * $q_Q = O(1)$ (fixed dimension $d_q$).
  * $q_{\mathrm{idx}} = \lceil \log_2 N \rceil$.
  * $q_{KV}, q_{\mathrm{out}} = O(1)$.

  Total: $O(\log N)$ qubits as a function of $N$.

* **Gate count per query.**

  * Query encoding: $O(1)$ gates in $N$.
  * Superposition over indices: $O(\log N)$ Hadamards.
  * QRAM-controlled key–value feature map: assumed $O(\mathrm{poly}(\log N))$ gates in $N$.
  * Aggregation and measurement: $O(1)$ in $N$.

  Under the QRAM cost model, per-query quantum cost is
  [
  O(\mathrm{poly}(\log N)).
  ]

* **Total attention-head cost.**

  We must process $N$ queries. Assuming queries are processed sequentially (no batching across queries on the quantum device),
  [
  T_{\mathrm{QLACI}}(N) = N \cdot O(\mathrm{poly}(\log N)) = O\big(N,\mathrm{poly}(\log N)\big)
  ]
  with respect to $N$.

This is strictly better than $O(N^2)$ and near-linear in $N$, matching the dependence of some efficient classical transformers up to polylogarithmic factors, but with a different structural origin: QLACI does not enforce locality or explicit sparsity in the attention pattern.

We emphasize that:

* Storage of $(K_j, V_j)$ still requires $O(N)$ classical or QRAM memory.
* The complexity claim is conditional on an efficient QRAM-like primitive; without QRAM, generic decomposition of multi-controlled accesses can reintroduce $O(N)$ or worse costs.

### 3.3. Hybrid Quantum–Classical Transformer Architecture

QLACI is used as a drop-in replacement for a single attention head in a transformer block, with the rest of the block classical. Figure 2 depicts the overall block structure.

#### 3.3.1. Block Structure

For each layer:

1. **Classical input.**

   * Tokens are embedded into $X \in \mathbb{R}^{N \times d}$.
   * Positional encodings are added.

2. **Classical projection.**

   * Linear maps produce $Q, K, V \in \mathbb{R}^{N \times d_q}$.

3. **QLACI attention.**

   * For each $i$, feed $Q_i$ into QLACI, with QRAM-backed access to $(K_j, V_j)$.
   * Collect context vectors $C = [c_1, \dots, c_N]^\top \in \mathbb{R}^{N \times d_{\mathrm{ctx}}}$.

4. **Residual and normalization.**

   * Apply a residual connection $X + W_{\mathrm{ctx}}C$ (with a learned projection $W_{\mathrm{ctx}}$) followed by layer normalization.

5. **Feed-forward network.**

   * Standard position-wise MLP:
     [
     \mathrm{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2.
     ]
   * Residual and layer normalization as usual.

The only quantum component is the QLACI subroutine in step 3.

#### 3.3.2. Training

* **Classical parameters.** Embeddings, projections, FFN weights, layer norms.
* **Quantum parameters.** Parameters $\theta,\phi$ in $U_{\mathrm{interact}}$ and $U_{\mathrm{score}}$.

Training loop:

1. **Forward pass.** Execute QLACI on a quantum simulator or hardware for each query, collect $C$, complete the classical forward pass, compute loss.
2. **Backward pass.**

   * For PQC parameters, use parameter-shift gradients [CITATION: Schuld et al., 2019] or simulator autodiff (e.g., Pennylane [CITATION: Bergholm et al., 2022]).
   * For classical parameters, use backpropagation.
3. **Optimization.** Update via Adam or similar optimizer [CITATION: Kingma & Ba, 2015].

Because quantum execution is costly, practical experiments will use small $N$ on simulators; the asymptotic analysis targets future hardware.

### 3.4. Hardware and Simulation Considerations

* **Hardware regime.** Near-term deployment will be limited by:

  * Qubit counts (tens to low hundreds).
  * Coherence times.
  * Two-qubit gate fidelities.

* **Indexing capacity.** In principle, $q_{\mathrm{idx}} = \log_2 N$ qubits index $N$ positions. This does not magically store $N$ key–value vectors; those must reside in a QRAM-like memory or classical device coupled to the processor. The advantage is in coherent addressing, not in bypassing memory requirements.

* **Simulators.** Experiments will primarily use:

  * Qiskit Aer [CITATION: Qiskit, 2023].
  * Pennylane [CITATION: Bergholm et al., 2022].
  * Cirq [CITATION: Cirq, 2023].

* **Noise.** For noisy simulations, depolarizing and readout error channels will be added to assess robustness. Error mitigation (e.g., zero-noise extrapolation, readout mitigation) can be explored but is not central to the conceptual contribution.

### 3.5. Practical Bottlenecks: QRAM, Trainability, Parallelism, and Measurement

The formal complexity analysis above isolates the $N$-dependence under optimistic assumptions. In practice, four bottlenecks largely determine whether any asymptotic advantage survives into hardware and wall-clock performance.

#### 3.5.1. QRAM Cost and Constant Factors

The QLACI scaling assumes a QRAM-like primitive with $\mathrm{poly}(\log N)$ access cost. Existing QRAM proposals (e.g., Giovannetti–Lloyd–Maccone bucket-brigade QRAM [CITATION: Giovannetti et al., 2008]) require $O(N)$ hardware size, long cascades of controlled operations, and have unfavorable error propagation. Other architectures rely on 3D cavities, atomic ensembles, or photonic schemes that are far beyond NISQ capabilities.

As a result:

* For the next few hardware generations, practical wall-clock time will likely be dominated by QRAM access cost, not the polylogarithmic PQC portion.
* Even in a fault-tolerant regime, constant factors from routing, error correction, and QRAM depth may delay the onset of any asymptotic advantage to very large $N$.

This paper therefore positions QLACI as an architecture for the *QRAM-available* regime, not as a proposal that yields speedups on current devices.

#### 3.5.2. Barren Plateaus and Structured Ansatz Design

QLACI uses a joint register of size $O(\log N)$ (query, index, ancilla, and output) with data-dependent controlled operations. Such architectures are known to be susceptible to barren plateaus: gradients vanishing exponentially in qubit count or depth [CITATION: McClean et al., 2018].

To mitigate this, we propose:

* **Shallow, layered ansätze** for $U_{\mathrm{interact}}$ and $U_{\mathrm{score}}$ with depth scaling at most logarithmically in $N$ and constant local connectivity.
* **Problem-structured parameter tying**, where parameters are shared across indices and layers to reduce effective dimension and avoid randomizing the full Hilbert space.
* **Data-reuploading strategies**, in which query information is injected at multiple points in the circuit rather than fully “scrambled” once, preserving gradient signal.
* **Initialization heuristics** that start near identity or near low-entropy states, avoiding maximally mixed effective states where gradients are provably flat.

These design constraints narrow the ansatz family compared to generic PQCs but are likely necessary for trainability.

#### 3.5.3. Per-Query Sequential Execution and Parallelism

In the simplest instantiation, QLACI processes the $N$ queries sequentially, because each requires encoding a different $Q_i$ on $R_Q$. This implies:

* Total runtime scales as $N \times (\text{QRAM} + \text{PQC} + \text{measurement})$.
* Quantum parallelism is used *within* each query over positions, but *not* across queries.

There are three possible mitigations:

1. **Multi-query encodings.** Introduce an additional query-index register and encode several queries in superposition, enabling a single circuit execution to produce context vectors for a mini-batch of queries at once. This trades more qubits and more complex decoding for fewer sequential calls.
2. **Data-parallel QPUs.** Execute QLACI independently for different query subsets on multiple QPUs, analogous to classical data parallelism.
3. **Hybrid schemes.** Use QLACI only for a subset of “global” queries (e.g., special tokens) while handling local queries with classical attention.

In this paper we analyze the simplest sequential regime for conceptual clarity; exploring multi-query QLACI variants is a natural extension.

#### 3.5.4. Measurement Overhead and Context Dimensionality

To obtain a $d_{\mathrm{ctx}}$-dimensional context vector via Pauli measurements, one typically needs to estimate $d_{\mathrm{obs}}$ expectation values, each requiring a number of shots that scales with the target precision. Naïvely, this yields an additional factor of $O(d_{\mathrm{ctx}} \times \text{shots})$ per query.

Two strategies reduce this overhead:

* **Compressed quantum context.** Choose $d_{\mathrm{ctx}}$ modest (e.g., tens), then expand back to the model dimension $d$ via a classical projection matrix. The quantum module then behaves as a learned low-dimensional “bottleneck” that captures global structure, with classical widening layers.
* **Grouped measurements and classical shadows.** Use measurement strategies that estimate several observables jointly or reconstruct approximate density matrices via classical shadows, reducing the number of distinct measurement settings.

Even with these techniques, measurement overhead remains a substantial constant factor in wall-clock time. In this work, it is treated explicitly as part of the per-query cost, not swept into asymptotic notation.

---

## 4. Experimental Setup

The experimental program is designed to answer two questions:

1. Do QLACI-based blocks exhibit empirical $O\big(N,\mathrm{poly}(\log N)\big)$ scaling in practice on simulators (as $N$ and qubit counts are varied)?
2. Do they remain competitive in task performance with strong classical baselines on long-sequence tasks?

### 4.1. Datasets

The focus is on tasks where long-range dependencies are known to matter.

#### 4.1.1. Long-Sequence Text

* **Long document summarization.**

  * PubMed and arXiv summarization datasets [CITATION: Cohan et al., 2018; Cohan et al., 2020].
* **Long-range QA.**

  * NarrativeQA [CITATION: Kočiský et al., 2018].
  * Qasper [CITATION: Das et al., 2021].
* **Synthetic tasks.**

  * Copying tasks where the relevant token lies far in the past.
  * Associative recall tasks with controlled distance between relevant items.

#### 4.1.2. Genomics

* DNA/RNA sequence modeling tasks (e.g., regulatory element prediction) using data from ENCODE [CITATION: ENCODE Project Consortium, 2012].
* Sequences tokenized as nucleotides or k-mers, with contexts up to tens or hundreds of kilobases.

#### 4.1.3. Long Time Series (Optional)

* Climate, finance, or industrial sensor datasets where context windows span years and include multiple periodicities.

For all datasets, sequence lengths will be systematically varied to probe scaling.

### 4.2. Baselines

#### 4.2.1. Standard Transformer

* Vanilla transformer architectures (e.g., BERT-style [CITATION: Devlin et al., 2019], GPT-style [CITATION: Radford et al., 2019]) adapted to each task.
* Context windows chosen as large as is feasible given hardware constraints.

#### 4.2.2. Efficient Transformers

* **Sparse:** Longformer [CITATION: Beltagy et al., 2020], BigBird [CITATION: Zaheer et al., 2020].
* **Linear/approximate:** Performer [CITATION: Choromanski et al., 2021], Reformer [CITATION: Kitaev et al., 2020], Linformer [CITATION: Wang et al., 2020].
* **Recurrent/memory:** Transformer-XL [CITATION: Dai et al., 2019], Compressive Transformer [CITATION: Rae et al., 2020].

These baselines define the classical accuracy–efficiency frontier for long sequences.

### 4.3. Metrics

#### 4.3.1. Task Performance

* **Summarization:** ROUGE-1/2/L [CITATION: Lin, 2004].
* **QA:** Exact Match and F1.
* **Language modeling:** Perplexity.
* **Genomics/time series:** Accuracy, F1, precision/recall, RMSE, MAE as appropriate.

#### 4.3.2. Computational Performance

For each model and sequence length $N$:

* Wall-clock training time per epoch (or per fixed number of updates).
* Inference time per example or batch.
* Peak memory usage.

Scaling curves will be plotted on log–log axes. For QLACI, the primary object of interest is the dependence of compute and memory on $N$, abstracting away constant factors associated with quantum simulation.

#### 4.3.3. Quantum Resource Metrics

For QLACI:

* Number of qubits as a function of $N$.
* Circuit depth and number of entangling gates.
* Number of PQC parameters.
* Number of shots used per measurement (for noisy simulations).

These metrics clarify hardware demands.

### 4.4. Protocol

* **Hyperparameter tuning.** Joint tuning for each model class on validation sets, including embedding sizes, number of layers, learning rates, and, for QLACI, PQC depth, ansatz structure, and shot counts.
* **Training regime.** Fixed training budgets or early stopping based on validation loss.
* **Environment.** Experiments run on GPUs for classical parts and CPUs or specialized hardware for quantum simulation backends.
* **Reproducibility.** Fixed random seeds, configuration dumps, and release of code.

### 4.5. Implementation Scope and Prototype Experiments

To ground the conceptual proposal, we provide a minimal open-source prototype that instantiates QLACI on a classical machine via state-vector simulation. The implementation is intentionally small-scale and should be read as an executable specification and sanity check, not as a competitive long-sequence model.

**Code.** The prototype consists of two Python scripts built with PyTorch, Pennylane (using the `lightning.qubit` backend), and Matplotlib:

1. `qlaci_real_n16.py` implements an 11-qubit QLACI-style PQC with 4 query qubits, 4 index qubits (supporting $N=16$ positions), and 3 ancilla/readout qubits. A synthetic “needle-in-a-haystack” regression task is constructed by embedding sequences of length $N=16$ and asking the model to reconstruct a hidden target vector from the final position conditioned on the past. The script trains QLACI end-to-end on this toy task and reports the mean-squared error (MSE) over training steps. The corresponding learning curve is shown in Figure 4 (“TRUE QLACI — N=16, Real Quantum Log-Attention; No pre-averaging, real index superposition”).

2. `qlaci_vs_classical_real.py` compares three models on the same synthetic task:

   * A **True QLACI** module using the 11-qubit circuit as the context aggregator.
   * A **Classical Linear Attention** baseline that applies a Performer-style positive feature map and computes linear-time attention over the past tokens.
   * A **Classical Average** baseline that simply averages past token embeddings.

   All three models share the same token embedding and output dimensions, differing only in the context aggregation mechanism. Training curves for 120 optimization steps are plotted in Figure 3 (“QLACI vs Classical Baselines — N=16; Real quantum log-attention with controlled QRAM simulation”).

**Figures.**

* **Figure 1** (QLACI Circuit Schematic) and the associated close-up highlight the query, index, key–value, and output registers, as well as the QRAM-mediated feature map and aggregation subcircuits.
* **Figure 2** (Hybrid Transformer Block Architecture) illustrates how a single QLACI head plugs into an otherwise standard transformer block, including residual and feed-forward pathways.
* **Figure 3** shows that, on the $N=16$ needle-recovery task, True QLACI, classical linear attention, and the average baseline all converge to similar MSE in the $\sim 10^0$ regime; QLACI tracks the linear baseline closely and consistently outperforms the naive average, but does not exhibit a clear accuracy advantage.
* **Figure 4** presents the QLACI learning curve alone, emphasizing that the PQC ansatz is trainable on this small-scale task and that the training dynamics are stable.

Because these experiments run entirely on classical simulators with a tiny number of qubits and a very limited sequence length, they provide **no evidence of quantum advantage**. Their sole purpose is to (i) verify that the QLACI ansatz admits usable gradients on a simple supervised problem, (ii) validate the correctness of the implementation, and (iii) demonstrate how the circuit-level primitive and the hybrid block of Section 3 can be instantiated in code.

---

## 5. Expected Results, Limitations, and Future Work

### 5.1. Scaling Behavior

We expect:

* For standard self-attention, empirical cost curves consistent with $O(N^2)$ in both compute and memory, becoming prohibitive at moderate $N$.
* For efficient classical transformers, near-linear empirical scaling with various constant factors and structural constraints.
* For QLACI-based blocks on simulators, cost dominated by the $N$ calls to a PQC whose depth grows $\mathrm{poly}(\log N)$, yielding empirical curves compatible with $O\big(N,\mathrm{poly}(\log N)\big)$ under the QRAM cost assumption.

Because simulators are themselves exponential in qubit count, actual experiments will be constrained to modest $N$ and will primarily validate the *trend* as qubit counts are varied, not the absolute asymptotic regime.

### 5.2. Task Performance

On shorter sequences where full classical attention is feasible, the hybrid model is expected to match or slightly lag strong classical baselines due to optimization difficulties and stochasticity in PQC outputs. The key question is functional viability: can QLACI-based blocks learn meaningful long-range dependencies at all?

On longer sequences beyond the comfortable regime of standard attention, the hybrid model is expected to:

* Outperform classical baselines that must truncate context.
* Compete with efficient classical transformers, potentially offering better handling of non-local dependencies because it does not hard-wire locality or sparsity patterns into the attention mechanism.

These expectations are conditional; empirical evidence is required.

### 5.3. Representational Analysis

If QLACI outputs can be mapped to interpretable “quantum attention maps” (e.g., via probing circuits or learned probes), one can:

* Visualize effective attention over positions as inferred from QLACI outputs.
* Study how context aggregation behaves as $N$ increases.
* Compare these patterns to classical attention maps to understand representational differences.

Synthetic tasks with controlled dependency distances will be particularly useful to characterize when QLACI succeeds or fails.

### 5.4. Limitations

Key limitations include:

* **QRAM dependence and hardware feasibility.** The claimed polylogarithmic quantum dependence on $N$ hinges on an efficient QRAM-like primitive. Large-scale, fault-tolerant QRAM remains hypothetical [CITATION: Aaronson, 2015]. Without it, the access cost to $(K_j,V_j)$ may dominate and erase the theoretical advantage, potentially for decades of hardware development.
* **NISQ constraints.** Noise, limited qubit counts, and connectivity restrictions will severely limit near-term $N$ for which QLACI can be instantiated on hardware. The proposal is mainly relevant to a future fault-tolerant or QRAM-capable regime.
* **Simulation overhead.** High-fidelity quantum simulation is exponentially costly in qubit count. Our experimental program is therefore restricted to small qubit regimes and primarily illustrative.
* **Training pathologies.** QLACI inherits barren plateau risks [CITATION: McClean et al., 2018]. The structured ansatz strategies in Section 3.5.2 are necessary but not guaranteed to avoid trainability issues.
* **Sequential queries and measurement cost.** In the basic design, queries are handled sequentially, and measurement overhead scales with the effective context dimension and required precision. Multi-query encodings and compressed context strategies (Section 3.5.3–3.5.4) can reduce wall-clock time but introduce additional architectural complexity.
* **Scope of current experiments.** The present prototype operates at $N=16$ on synthetic data, with all quantum computation emulated on a classical simulator. The results demonstrate correctness and trainability, but they do not probe regimes where classical $O(N^2)$ attention is infeasible or where quantum hardware characteristics meaningfully affect performance.

### 5.5. Future Directions

Several directions follow:

* **Alternative quantum attention primitives.** Explore designs based on quantum singular value transformations, quantum associative memory, or other Hamiltonian-based aggregation schemes with different complexity–expressivity trade-offs.
* **Hardware-aware ansätze.** Tailor QLACI circuits to specific device topologies and noise profiles, reducing depth and gate count to match realistic NISQ constraints and actively monitoring barren plateau indicators.
* **Multi-query QLACI.** Develop and analyze QLACI variants that process multiple queries in superposition or via parallel QPUs, closing the gap between asymptotic per-query complexity and wall-clock throughput.
* **Beyond attention.** Quantum treatments of feed-forward layers, embeddings, or output heads, to understand where quantum computation offers the most leverage in transformer-style models.
* **Complexity-theoretic analysis.** Sharpen the understanding of what, if anything, QLACI-like primitives can guarantee beyond classical near-linear attention, under realistic models of quantum and classical hardware, including QRAM cost and error-correction overhead.
* **Larger-scale simulations and benchmarks.** As simulators and hardware improve, extend experiments to longer sequences, richer tasks (e.g., long-document summarization, genomics), and more realistic transformer backbones.

In summary, this work does not claim a proven, unconditional quantum speedup over all classical efficient transformers. It provides a concrete, explicitly assumption-labeled architecture in which a quantum subroutine plays the role of a global attention mechanism with polylogarithmic dependence on sequence length in its quantum core, together with a small but executable prototype, and it outlines how such a mechanism might be evaluated empirically as quantum hardware and memory models mature.
