# Gollek Examples

A comprehensive collection of working examples demonstrating Gollek's capabilities across JBang scripts and Jupyter notebooks.

## 📚 Example Categories

### 🎨 **Multimodal AI** (Image, Audio, Video)

#### JBang Examples
- [stable_diffusion_generation.java](jbang/multimodal/stable_diffusion_generation.java) - Text-to-image generation with Stable Diffusion ✨ NEW
- [stable_diffusion_batch.java](jbang/multimodal/stable_diffusion_batch.java) - Batch generate multiple variations ✨ NEW
- [stable_diffusion_metal.java](jbang/multimodal/stable_diffusion_metal.java) - SD with Metal acceleration (Apple Silicon)
- [text_to_speech.java](jbang/multimodal/text_to_speech.java) - Text-to-Speech with SpeechT5 ✨ NEW
- [vision_storyteller.java](jbang/multimodal/vision_storyteller.java) - Image-to-text with LLaVA
- [native_whisper_stt.java](jbang/multimodal/native_whisper_stt.java) - Speech-to-Text with Whisper
- [smart_transcriber.java](jbang/multimodal/smart_transcriber.java) - Audio transcription + highlights
- [video_analyst.java](jbang/multimodal/video_analyst.java) - Video temporal analysis
- [omni_assistant.java](jbang/multimodal/omni_assistant.java) - Mixed-modality reasoning

#### Jupyter Notebooks
- [07-stable-diffusion.ipynb](jupyter/07-stable-diffusion.ipynb) - Interactive SD generation with visualization ✨ NEW

### 🗣️ **Natural Language Processing**

#### JBang Examples
- [nlp_chat_qwen_gguf.java](jbang/nlp/nlp_chat_qwen_gguf.java) - Chat with Qwen GGUF model
- [nlp_concurrent_batch_pipeline.java](jbang/nlp/nlp_concurrent_batch_pipeline.java) - High-throughput batching
- [nlp_high_performance.java](jbang/nlp/nlp_high_performance.java) - Optimized text generation
- [nlp_semantic_search_simd.java](jbang/nlp/nlp_semantic_search_simd.java) - Vector similarity search
- [nlp_sentiment_analysis.java](jbang/nlp/nlp_sentiment_analysis.java) - Text classification
- [nlp_transformer_classifier.java](jbang/nlp/nlp_transformer_classifier.java) - Custom classifier training

### 🔧 **SDK Integration**

#### JBang Examples
- [gollek-quickstart.java](jbang/sdk/gollek-quickstart.java) - Quick start guide
- [gollek-sdk-core-example.java](jbang/sdk/gollek-sdk-core-example.java) - Core SDK features
- [gollek-sdk-vision-example.java](jbang/sdk/gollek-sdk-vision-example.java) - Vision API
- [gollek-sdk-train-example.java](jbang/sdk/gollek-sdk-train-example.java) - Training API
- [gollek-sdk-augment-example.java](jbang/sdk/gollek-sdk-augment-example.java) - Data augmentation
- [gollek-sdk-export-example.java](jbang/sdk/gollek-sdk-export-example.java) - Model export
- [graph_fusion_example.java](jbang/sdk/graph_fusion_example.java) - Graph optimization
- [unified_framework_demo.java](jbang/sdk/unified_framework_demo.java) - Unified inference
- [mnist_training_v02.java](jbang/sdk/mnist_training_v02.java) - MNIST training
- [pytorch_comparison_v02.java](jbang/sdk/pytorch_comparison_v02.java) - PyTorch comparison
- [tensor_operations_v02.java](jbang/sdk/tensor_operations_v02.java) - Tensor math
- [tokenization_v02.java](jbang/sdk/tokenization_v02.java) - Tokenizer usage
- [vision_transforms_v02.java](jbang/sdk/vision_transforms_v02.java) - Image transforms

### ⚡ **Edge Inference (LiteRT)**

#### JBang Examples
- [LiteRTEdgeExample.java](jbang/edge/LiteRTEdgeExample.java) - Basic LiteRT usage
- [ImageClassificationExample.java](jbang/edge/ImageClassificationExample.java) - Image classification
- [ObjectDetectionExample.java](jbang/edge/ObjectDetectionExample.java) - Object detection
- [TextGenerationExample.java](jbang/edge/TextGenerationExample.java) - On-device text generation
- [BenchmarkExample.java](jbang/edge/BenchmarkExample.java) - Performance benchmarking
- [LiteRTMetalBridgeExample.java](jbang/edge/LiteRTMetalBridgeExample.java) - Metal acceleration
- [LiteRTGemmaExample.java](jbang/edge/LiteRTGemmaExample.java) - Gemma on edge
- [LiteRTGemmaMetalExample.java](jbang/edge/LiteRTGemmaMetalExample.java) - Gemma + Metal

### 🔢 **Quantization**

#### JBang Examples
- [gollek-quantizer-gptq.java](jbang/quantizer/gollek-quantizer-gptq.java) - GPTQ quantization
- [gollek-quantizer-awq.java](jbang/quantizer/gollek-quantizer-awq.java) - AWQ quantization
- [gollek-quantizer-autoround.java](jbang/quantizer/gollek-quantizer-autoround.java) - AutoRound quantization
- [gollek-quantizer-turboquant.java](jbang/quantizer/gollek-quantizer-turboquant.java) - TurboQuant (INT4/INT8/FP8)
- [gollek-quantizer-comparison.java](jbang/quantizer/gollek-quantizer-comparison.java) - Method comparison

### 🧠 **Common/Basic**

#### JBang Examples
- [hello_gollek.java](jbang/common/hello_gollek.java) - Hello World
- [error_handling.java](jbang/common/error_handling.java) - Error handling patterns
- [batch_process.java](jbang/common/batch_process.java) - Batch processing
- [custom_module_demo.java](jbang/common/custom_module_demo.java) - Custom modules
- [mnist_style_setup.java](jbang/common/mnist_style_setup.java) - Simple training
- [model_persistence.java](jbang/common/model_persistence.java) - Save/load models
- [model_persistence_safetensor.java](jbang/common/model_persistence_safetensor.java) - SafeTensors format
- [train_cli.java](jbang/common/train_cli.java) - Training CLI
- [train_model.java](jbang/common/train_model.java) - Model training

### 🔌 **Third-Party Integrations**

#### JBang Examples
- [deeplearning4j_integration.java](jbang/integration/deeplearning4j_integration.java) - DL4J integration
- [opennlp_integration.java](jbang/integration/opennlp_integration.java) - Apache OpenNLP
- [smile_ml_integration.java](jbang/integration/smile_ml_integration.java) - Smile ML library
- [stanford_nlp_integration.java](jbang/integration/stanford_nlp_integration.java) - Stanford NLP
- [tribuo_integration.java](jbang/integration/tribuo_integration.java) - Oracle Tribuo

### 🧪 **Neural Networks & ML**

#### JBang Examples
- [neural_network_with_gollek.java](jbang/neural_network/neural_network_with_gollek.java) - Custom NN
- [neural-network-example.java](jbang/neural_network/neural-network-example.java) - Simple NN
- [cv_mlp_classifier_export.java](jbang/machine_learning/cv_mlp_classifier_export.java) - MLP classifier
- [simd_audio_processing.java](jbang/machine_learning/simd_audio_processing.java) - SIMD audio

## 🚀 Quick Start

### JBang (Command-Line)

```bash
# Install JBang first
curl -Ls https://sh.jbang.dev | bash -s - app setup

# Run any example
jbang examples/jbang/multimodal/stable_diffusion_generation.java

# With custom parameters
jbang examples/jbang/multimodal/stable_diffusion_generation.java \
  --prompt "a cyberpunk cat" \
  --output cyberpunk.png \
  --steps 30
```

### Jupyter Notebook (Interactive)

```bash
# Install Java kernel
jbang app install java-kernel

# Register kernel
jupyter kernelspec install examples/jupyter/kernel.json

# Launch Jupyter
jupyter notebook examples/jupyter

# Open 07-stable-diffusion.ipynb for image generation
```

## 📊 Feature Matrix

| Category | JBang | Jupyter | Description |
|----------|-------|---------|-------------|
| **Text-to-Image** | ✅ 3 | ✅ 1 | Stable Diffusion generation |
| **Text-to-Speech** | ✅ 1 | ❌ | SpeechT5 synthesis |
| **Speech-to-Text** | ✅ 2 | ❌ | Whisper transcription |
| **Vision QA** | ✅ 1 | ❌ | LLaVA image understanding |
| **Video Analysis** | ✅ 1 | ❌ | Temporal video understanding |
| **NLP/Chat** | ✅ 6 | ✅ 1 | Text generation & classification |
| **SDK Usage** | ✅ 13 | ✅ 1 | Core API examples |
| **Edge/LiteRT** | ✅ 9 | ❌ | Mobile/edge inference |
| **Quantization** | ✅ 5 | ❌ | Model compression |
| **Integrations** | ✅ 5 | ❌ | Third-party libraries |

**Total**: 53+ JBang scripts, 2+ Jupyter notebooks

## 🎯 Learning Paths

### Beginner Path
1. `hello_gollek.java` - Basic usage
2. `01-getting-started.ipynb` - Interactive tensors
3. `mnist_style_setup.java` - Simple training
4. `gollek-quickstart.java` - SDK basics

### Intermediate Path
1. `stable_diffusion_generation.java` - Image generation ✨ NEW
2. `native_whisper_stt.java` - Audio processing
3. `gollek-sdk-vision-example.java` - Computer vision
4. `nlp_sentiment_analysis.java` - NLP basics

### Advanced Path
1. `omni_assistant.java` - Multimodal AI
2. `gollek-quantizer-turboquant.java` - Quantization
3. `video_analyst.java` - Video understanding
4. `graph_fusion_example.java` - Optimization

## 📝 Creating Your Own Examples

### Converting Jupyter → JBang

1. Extract code cells from `.ipynb`
2. Add JBang header with dependencies
3. Add CLI argument parsing
4. Run with `jbang your_script.java`

### Converting JBang → Jupyter

1. Remove JBang header
2. Split into logical code cells
3. Add markdown explanation cells
4. Save as `.ipynb`

## 🐛 Troubleshooting

### JBang Issues

**Problem**: `Error: Could not find or load main class`
**Solution**: Ensure Java 25 is installed: `java -version`

**Problem**: Dependencies not resolving
**Solution**: Run `jbang cache clear` and retry

### Jupyter Issues

**Problem**: Kernel not found
**Solution**: Install Java kernel: `jbang app install java-kernel`

**Problem**: Imports failing
**Solution**: Ensure Gollek JARs are in kernel classpath

## 🤝 Contributing

We welcome new examples! Please follow these guidelines:

1. **Working Code**: Examples must compile and run
2. **Clear Comments**: Explain what each section does
3. **CLI Arguments**: Support `--help` and custom parameters
4. **Error Handling**: Show best practices for error handling
5. **Documentation**: Update this INDEX.md when adding new examples

## 📚 Additional Resources

- [Stable Diffusion Tutorial](https://gollek-ai.github.io/docs/tutorials/intermediate/stable-diffusion/)
- [API Reference](https://gollek-ai.github.io/docs/api-references/0.1.0-SNAPSHOT/modules/)
- [CLI Documentation](https://gollek-ai.github.io/docs/cli/)
- [Framework Guides](https://gollek-ai.github.io/docs/framework/)

---

**Last Updated**: April 14, 2026  
**Total Examples**: 53+ JBang, 2+ Jupyter
