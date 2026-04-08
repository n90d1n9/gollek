# Core-Step-25: Mixture-of-Experts Runtime

## ❗ The Problem
Dense models (where every parameter is used for every token) hit a scaling wall. 1T+ parameter models are too slow and expensive for real-time use. Mixture-of-Experts (MoE) solves this by only activating a small fraction (e.g. 2 of 16 experts) per token, but MoE introduces new challenges:
1. **Expert Bottlenecks**: If all tokens route to "Expert 1", that node becomes a hotspot.
2. **High VRAM Overhead**: Storing 16 experts (even if only 2 are used) requires massive memory.
3. **Complex Routing**: The control plane needs to know exactly where each expert lives in a distributed cluster.

---

## ✅ The Solution: Sparse-Aware Fabric
Gollek is now natively optimized for MoE architectures.

### 1. Distributed Expert Gating
The `ExpertRouter` SPI handles Top-K selection. The gating logic is no longer hidden in the runner; it's a first-class citizen of the inference pipeline.
*   **Expert Locations**: The `KVDirectory` now tracks exactly which nodes host which experts.
*   **Remote Expert Execution**: If an expert is not local, the activation tensor is routed to the remote node, processed, and returned (RPC-based MoE).

### 2. Expert-Aware Memory Management
*   `prefetchExperts()`: Proactively loads weights into GPU memory based on predicted routing distributions.
*   **Affinity Loading**: Groups "related" experts together on the same node to minimize inter-node hops.

### 3. Sparse Telemetry
*   **Expert Attribution**: Every token trace now includes which experts were used.
*   **Heatmaps**: Real-time visualization of expert load, enabling the scheduler to re-balance "hot" experts.

---

## 🚀 The Result: The 1T-Parameter Edge
Gollek can now run models that were previously reserved for massive data centers. By sharding experts across a home cluster or a small enterprise farm, we achieve **Data Center Class Intelligence** on **Commodity Hardware**.