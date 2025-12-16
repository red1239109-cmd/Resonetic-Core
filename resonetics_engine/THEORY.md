# 📐 Theoretical Background: The Resonetics Engine

The Resonetics engine was designed to mitigate the **"logical hallucinations"** that plague standard large language models (LLMs). We look beyond semantics—at **structure**.

## 1. Core Problem: The Limits of Cosine Similarity

Conventional AIs rely on **cosine similarity** in vector space.
* **Issue:** "A is B" and "A is not B" share almost the same words, so their vectors sit extremely close together.
* **Result:** Even when a logical contradiction exists, the model believes the sentences are "semantically similar" and happily produces a plausible-sounding falsehood.

## 2. Solution: Topological Data Analysis (TDA)

We treat a text embedding not as a single point but as a **trajectory through time**.

### 2.1 Time-Delay Embedding (Takens' Theorem)
$$\mathbf{X}_i = [x_i, x_{i+\tau}, x_{i+2\tau}, \dots, x_{i+(m-1)\tau}]$$
By sliding a window across the sentence vector we reconstruct a **point cloud in a high-dimensional phase space**, revealing hidden structural patterns inside the sentence.

### 2.2 Persistent Homology
We build a **Vietoris–Rips complex** on that cloud and extract topological features.
* **$H_0$ (Components):** connected components; indicate semantic cohesion.
* **$H_1$ (Loops/Holes):** **logical "holes."** A strong $H_1$ signal between premise and conclusion means the argument jumps instead of flowing—an inferential gap.

### 2.3 Bottleneck Distance
We measure the distance between the two resulting persistence diagrams:
$$W_\infty(D_1, D_2) = \inf_\gamma \sup_{x \in D_1} \|x - \gamma(x)\|_\infty$$
* **Distance ≈ 0:** the logical structure is perfectly preserved (resonance).
* **Distance ≫ 0:** a structural mismatch occurs (dissonance/shock).

## 3. Hybrid Inference Architecture (v6.1)

TDA is cubic in the worst case. We keep it real-time with a three-tier strategy:

1. **Cache Hit:** already-computed diagrams are served in $O(1)$ by `SizedLRUCache`.
2. **Async TDA:** new material is refined in a separate process without blocking the stream.
3. **Fallback:** if computation exceeds the `timeout` or data are insufficient, we switch to a cosine-similarity approximation to guarantee availability.

---
*References:*
* *Carlsson, G. (2009). "Topology and Data". Bulletin of the AMS.*
* *Reimers, N., & Gurevych, I. (2019). "Sentence-BERT". EMNLP.*
