# E3VB Codex Task Specification

## 1. Project Goal

The goal of this project is to build an end-to-end model for **predicting Valence Bond (VB) structure weights**.

The target architecture is:

- **E3nn-style atom-level encoder** for learning geometry-aware atom representations
- **Rumer / orbital-level module** for learning orbital and VB-structure-specific interactions
- an end-to-end training pipeline in which both the atom-level and orbital-level parts are jointly optimized

In other words:

- the **E3nn part** is responsible for training the **atom-level representation**
- the **Rumer part** is responsible for training the **orbital-level / VB-structure-level representation**
- the final model must predict **VB structure weights**
- the final scalar output must satisfy **strict E(3) invariance**
- if the implementation uses parity-aware O(3) semantics, then the final scalar output must also satisfy **reflection invariance**

This is the most important objective of the project.

---

## 2. Core Technical Requirements

The implementation must strictly follow these requirements:

- Use **Python**
- Use **JAX** for numerical computation
- Use **Flax NNX** as the neural network framework
- Use **Grain** for the data pipeline
- Use **Jraph** for graph data structures / graph processing where appropriate

The implementation must **not** use:

- PyTorch
- TensorFlow
- Haiku
- Flax Linen

Important notes:

- The project should be implemented as a **native JAX project**
- Do not wrap old PyTorch logic as the main solution
- Do not keep PyTorch-based model execution in the new training path
- Do not replace equivariant modeling with an ordinary non-equivariant GNN approximation

---

## 3. Scientific and Architectural Objective

The new project must preserve the scientific meaning of the current E3nn + Rumer workflow while rewriting it into a clean end-to-end architecture.

The desired forward pipeline is:

1. molecular geometry enters the atom-level encoder
2. atom-level features are learned from geometry
3. atom features are mapped into orbital-level features
4. orbital / Rumer graph message passing is performed
5. the final model predicts VB structure weights

The new implementation must satisfy the following:

- no offline atom feature export is required for the main training workflow
- the atom encoder and orbital / Rumer encoder must be trained jointly
- the atom-to-orbital bridge must be part of the trainable model pipeline
- the final prediction must be produced from a structurally invariant scalar path

---

## 4. Invariance Requirement

This is a hard requirement.

The final model output must satisfy **strict E(3) invariance**.

That means:

- translation invariance
- rotation invariance

If the implementation uses parity-aware O(3) semantics, it must also satisfy:

- reflection invariance

These properties must be guaranteed by the **model structure**, not only approximated by training.

The implementation must therefore ensure that:

- equivariant geometric processing follows valid irreps / tensor product rules
- the final scalar prediction is read out only from valid invariant scalar channels
- no non-equivariant shortcut path is allowed to influence the final prediction
- atom-to-orbital projection and final Rumer readout must not break invariance semantics

Numerical invariance tests must be provided and should show errors close to machine precision rather than structural errors at the `1e-3` to `1e-2` scale.

---

## 5. Data and Graph Responsibilities

The project contains two conceptual graph levels:

### Atom graph
The atom graph is responsible for:

- molecular geometry processing
- atom-level message passing
- learning atom representations from coordinates and atomic information

The atom graph should perform **dynamic geometric message construction** during forward, including quantities such as distances, relative vectors, radial features, angular features, and other geometry-dependent message terms.

### Orbital / Rumer graph
The orbital / Rumer graph is responsible for:

- orbital-level reasoning
- inactive / active orbital semantics
- VB-structure-specific pairing patterns
- structure-level prediction

The orbital / Rumer graph topology should be treated as **static metadata** prepared during preprocessing.

Examples of static orbital metadata include:

- `orb2atom`
- `orb_role_id`
- active orbital identities
- VB-structure-specific active connectivity / pairing

`.xmo` parsing belongs to preprocessing and data construction, not to model forward execution.

---

## 6. Unified Data Processing Requirement

The atom-level input pipeline and the orbital-level input pipeline currently have substantial overlap.

After refactoring, the project should avoid maintaining two disconnected preprocessing flows that repeatedly parse or reconstruct overlapping information.

The preferred design is:

- a unified data processing entry point or main sample builder
- one coherent sample object containing:
  - atom-level graph inputs
  - orbital metadata
  - Rumer graph topology information
  - labels and normalization-related metadata

This does **not** mean putting everything into one giant script.

It means the data flow should be unified and non-redundant.

The design should clearly distinguish:

- information used by the atom graph
- static prior metadata used by the orbital / Rumer graph
- shared underlying molecular information used by both views

---

## 7. Active Orbital Slot Matching Requirement

If multiple active orbitals belong to the same atom, they must not be treated as a single indistinguishable projection target.

The implementation should include a trainable slot-based active orbital matching mechanism.

For atom `i`, active slot `k`, and local orbital candidate `m`:

\[
u_{i,k}^{(0)} = \mathrm{MLP}([h_i^{0e}, \mathrm{Emb}(k)])
\]

\[
a_{i,k,m} = \mathrm{MLP}([u_{i,k}, \phi_{i,m}])
\]

\[
\alpha_{i,k,m} = \mathrm{softmax}_m(a_{i,k,m})
\]

\[
x_{i,k}^{\mathrm{active}} = \sum_m \alpha_{i,k,m} \phi_{i,m}
\]

Where:

- \(h_i^{0e}\) is the invariant scalar atom feature for atom `i`
- \(\mathrm{Emb}(k)\) is the learnable slot embedding
- \(\phi_{i,m}\) is the feature of local orbital candidate `m` on atom `i`
- \(x_{i,k}^{\mathrm{active}}\) is the active slot representation used for downstream orbital / Rumer reasoning

This mechanism should be treated as a formal trainable module in the end-to-end model, not as an offline auxiliary script.

---

## 8. Code Organization Requirements

The implementation must be modular and primarily organized around:

```text
E3nn/E3VB/
  main.py
  data/
  model/
  utils/

### Special Emphasis on Angular Features

For this project, the angular feature path must remain consistent with the original `E3nnVB/VB` design:

- use `lmax=1`
- use angular irreps `0e + 1o`
- explicitly keep the `1o` channel
- propagate the `1o` information into orbital features

Do not replace this with an `lmax=2` path or a scalar-only (`0e` only) simplification unless explicitly approved.