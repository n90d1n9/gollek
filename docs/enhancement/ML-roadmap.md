# Gollek ML SDK: Roadmap to PyTorch Parity for NLP/ML

While the current Gollek ML SDK establishes a robust "define-by-run" autograd engine and neural network module system (Linear, Embedding, Transformers, AdamW), achieving true feature-parity with PyTorch for advanced ML and NLP workloads requires addressing several key gaps.

## 1. NLP Tokenization ecosystem (Critical)
**Current State:** The `gollek-sdk-nlp` module pipelines handle inference nicely, but there is no public API for tokenization to *train* models.
**Improvement:** 
- Implement a `Tokenizer` interface (`encode`, `decode`, `batchEncode`).
- Create loaders for HuggingFace `tokenizer.json` (BPE, WordPiece, SentencePiece) so developers can easily tokenize datasets natively in Java before passing them to the `DataLoader` for training.

## 2. Model Serialization / StateDict (Critical)
**Current State:** `ModelHub` can load SafeTensors into a `Module`, but we lack the ability to save a newly trained model to disk.
**Improvement:** 
- Implement a `StateDict` API (equivalent to PyTorch's `model.state_dict()`).
- Add `Module.save(Path)` and `Module.load(Path)` methods that serialize parameters in the widely-adopted `.safetensors` format, ensuring cross-compatibility with Python ecosystems.

## 3. Tensor Slicing & Advanced Operations
**Current State:** `GradTensor` supports basic reshaping, flattening, and standard arithmetic, but lacks advanced indexing manipulation.
**Improvement:** 
- Add `slice(dim, start, end)` to fetch sequences or batches. 
- Add `cat` (concatenate) and `stack` along specific dimensions.
- Add `gather` and `scatter` ops, which are heavily used in custom loss functions and attention mechanisms.

## 4. Hardware Acceleration (Native Offloading)
**Current State:** The `GradTensor` internally uses `float[]` arrays, which means all autograd and forward passes execute on the CPU.
**Improvement:**
- Bind the `GradTensor` storage to the actual Gollek C++ core (which supports CUDA/Metal via the kernel plugins). 
- Introduce `tensor.to(Device.CUDA)` semantics to transparently move memory buffers to the GPU for training, matching PyTorch's device placement.

## 5. Vision and Multimodal Extensions (Future)
**Current State:** Focused entirely on standard feed-forward and text.
**Improvement:**
- Implement `Conv1d`, `Conv2d`, `Conv3d` and pooling layers (`MaxPool2d`).
- Integrate Java's `BufferedImage` with the `Dataset` API for vision transforms (`Resize`, `Normalize`, `RandomCrop`).

---

### Immediate Action Plan

To ensure the SDK is truly viable for NLP training, we should implement the **Top 2 Critical Gaps** immediately:
1. **Model Saving (`StateDict`)**: So users can save the models they train.
2. **Tensor Manipulation Ops**: Add `concat`, `stack`, and `slice` to `GradTensor` to allow for proper batching and sequence handling.
