# Gollek SDK Integration Examples

This directory contains comprehensive examples demonstrating how to integrate the Gollek SDK with popular Java ML/NLP libraries.

## Available Integrations

### 1. Deeplearning4j Integration
**File**: `examples/integration/deeplearning4j_integration.java`

Integrates Gollek SDK with [Deeplearning4j](https://deeplearning4j.konduit.ai/), a comprehensive deep learning library for the JVM.

**Use Cases**:
- Using DL4J pre-trained models (ResNet, VGG, etc.) alongside Gollek models
- Converting between DL4J INDArrays and Gollek tensors
- Combining DL4J's image preprocessing pipeline with Gollek inference
- Hybrid workflows leveraging both frameworks' strengths

**Run**:
```bash
jbang examples/integration/deeplearning4j_integration.java
```

### 2. Stanford NLP Integration
**File**: `examples/integration/stanford_nlp_integration.java`

Integrates Gollek SDK with [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/), a leading natural language processing toolkit.

**Use Cases**:
- Using Stanford NLP for advanced text preprocessing (tokenization, POS tagging, NER)
- Feeding linguistically-rich features into Gollek models
- Combining Stanford sentiment analysis with Gollek predictions
- Building multi-stage NLP pipelines

**Run**:
```bash
jbang examples/integration/stanford_nlp_integration.java
```

### 3. Apache OpenNLP Integration
**File**: `examples/integration/opennlp_integration.java`

Integrates Gollek SDK with [Apache OpenNLP](https://opennlp.apache.org/), a machine learning based toolkit for processing natural language text.

**Use Cases**:
- Text preprocessing and feature extraction
- Named entity recognition
- Sentence detection and tokenization
- Combining OpenNLP tools with Gollek inference

**Run**:
```bash
jbang examples/integration/opennlp_integration.java
```

### 4. Smile ML Integration
**File**: `examples/integration/smile_ml_integration.java`

Integrates Gollek SDK with [Smile](https://haifengl.github.io/) (Statistical Machine Intelligence & Learning Engine), a fast and comprehensive machine learning system.

**Use Cases**:
- Using Smile's data preprocessing and normalization
- Combining Smile's traditional ML algorithms with Gollek neural networks
- Feature engineering with Smile, inference with Gollek
- Ensemble methods combining both libraries

**Run**:
```bash
jbang examples/integration/smile_ml_integration.java
```

### 5. Oracle Tribuo Integration
**File**: `examples/integration/tribuo_integration.java`

Integrates Gollek SDK with [Oracle Tribuo](https://tribuo.org/), a machine learning library providing classification, regression, clustering, and more.

**Use Cases**:
- Using Tribuo's feature extraction pipelines
- Combining Tribuo classifiers with Gollek models
- Ensemble methods using both libraries
- Multi-model prediction aggregation

**Run**:
```bash
jbang examples/integration/tribuo_integration.java
```

### 6. Multimodal Integration (Native)
 **Directory**: `examples/multimodal/`
 
 Demonstrates Gollek's native multimodal processing capabilities (Vision, Audio, Video) using the high-level fluent API.
 
 **Use Cases**:
 - **Vision Storytelling**: Image-to-Text generation.
 - **Smart Transcription**: Audio-to-Text with summarization.
 - **Video Analysis**: Temporal scene understanding.
 - **Omni Assistant**: Cross-modality reasoning.
 
 **Run**:
 ```bash
 # Vision
 jbang examples/multimodal/vision_storyteller.java [image]
 
 # Audio
 jbang examples/multimodal/smart_transcriber.java [audio]
 
 # Video
 jbang examples/multimodal/video_analyst.java [video]
 
 # Omni
 jbang examples/multimodal/omni_assistant.java [image] [audio] [prompt]
 ```
 
 ## Common Integration Patterns

### Pattern 1: Data Conversion

Most integrations require converting between different tensor/data formats:

```java
// DL4J INDArray → Gollek Tensor
INDArray dl4jArray = ...;
float[] data = dl4jArray.toFloatVector();
GradTensor tensor = GradTensor.of(data, ...);

// Gollek Tensor → DL4J INDArray
GradTensor tensor = ...;
INDArray dl4jArray = Nd4j.create(tensor.data(), tensor.shape());
```

### Pattern 2: Pipeline Composition

Chain preprocessing from one library with inference from another:

```java
// Stanford NLP preprocessing → Gollek inference
String text = "...";
List<CoreLabel> tokens = stanfordTokenizer.tokenize(text);
String[] features = extractFeatures(tokens);
float[] vector = vectorize(features);
GradTensor input = GradTensor.of(vector, ...);
GradTensor output = gollekModel.forward(input);
```

### Pattern 3: Ensemble Predictions

Combine predictions from multiple libraries:

```java
// Get predictions from both libraries
float dl4jScore = dl4jModel.predict(input);
float gollekScore = gollekModel.predict(input);

// Ensemble: weighted average
float ensembleScore = 0.6f * dl4jScore + 0.4f * gollekScore;
```

### Pattern 4: Feature Fusion

Concatenate features from different sources:

```java
// Extract features from multiple sources
float[] dl4jFeatures = dl4jExtractor.extract(image);
float[] gollekFeatures = gollekExtractor.extract(image);

// Concatenate and feed to classifier
float[] combined = concat(dl4jFeatures, gollekFeatures);
GradTensor input = GradTensor.of(combined, 1, combined.length);
```

## Prerequisites

### Java Version
- Java 21 or higher required for all examples

### jbang Setup
All examples use [jbang](https://jbang.dev/) for dependency management:

```bash
# Install jbang
curl -Ls https://sh.jbang.dev | bash -

# Verify installation
jbang version
```

### Network Access
First run will download dependencies (may take 1-2 minutes):
- Maven Central repositories
- Library-specific repositories (e.g., Sonatype for Stanford NLP)

## Running the Examples

### Basic Execution

```bash
cd gollek/sdk/integration/jbang-templates/examples/integration
jbang deeplearning4j_integration.java
```

### With Custom Arguments

Some examples support command-line arguments:

```bash
jbang examples/integration/stanford_nlp_integration.java --model=sentiment --text="Great movie!"
```

### Export as Standalone JAR

```bash
jbang -o integration.jar examples/integration/deeplearning4j_integration.java
java -jar integration.jar
```

## Troubleshooting

### Slow First Run
**Normal behavior** - jbang downloads and caches dependencies.

**Solution**: Subsequent runs are fast. For offline use:
```bash
jbang cache deps examples/integration/deeplearning4j_integration.java
```

### Memory Issues
Some libraries (especially DL4J and Stanford NLP) require significant memory.

**Solution**: Add JAVA_OPTIONS in the script or use:
```bash
JAVA_OPTS="-Xmx4g" jbang deeplearning4j_integration.java
```

### Native Library Errors
Some libraries use native code (ND4J, etc.).

**Solution**: Ensure `--enable-native-access=ALL-UNNAMED` is in JAVA_OPTIONS (already included in examples).

### Dependency Conflicts
If you encounter version conflicts:

**Solution**: Clear jbang cache:
```bash
jbang cache clear
```

## Integration Best Practices

### 1. Resource Management
Always close resources properly:

```java
try (var model = loadModel()) {
    // use model
} finally {
    model.close();
}
```

### 2. Thread Safety
Most ML libraries are not thread-safe by default. Use proper synchronization:

```java
synchronized(model) {
    model.predict(input);
}
```

### 3. Memory Efficiency
Release unused tensors/arrays:

```java
// DL4J
Nd4j.getMemoryManager().garbageCollect();

// Gollek
tensor.detach();
```

### 4. Error Handling
Wrap external library calls in try-catch:

```java
try {
    // External library call
} catch (ExternalLibraryException e) {
    // Handle gracefully
    logger.warn("External library failed, using fallback", e);
}
```

## Performance Considerations

### Warm-up
First inference is slow due to JIT compilation:

```java
// Warm-up
for (int i = 0; i < 3; i++) {
    model.forward(dummyInput);
}
```

### Batching
Process multiple inputs together for better throughput:

```java
// Batch processing
GradTensor batch = GradTensor.of(batchData, batchSize, features);
GradTensor outputs = model.forward(batch);
```

### Model Caching
Cache loaded models for reuse:

```java
static final Model CACHED_MODEL = loadModel();

static {
    // Pre-warm
    CACHED_MODEL.forward(warmupInput);
}
```

## Contributing

When adding new integration examples:

1. Follow the existing naming convention: `{library}_integration.java`
2. Include comprehensive comments
3. Show at least 2-3 use cases
4. Include error handling
5. Add to this README
6. Test with `jbang {file}.java`

## Additional Resources

- [Gollek SDK Documentation](../../../README.md)
- [Deeplearning4j Documentation](https://deeplearning4j.konduit.ai/)
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- [Apache OpenNLP](https://opennlp.apache.org/)
- [Smile ML](https://haifengl.github.io/)
- [Oracle Tribuo](https://tribuo.org/)
- [jbang Documentation](https://jbang.dev/documentation)

## License

These examples are provided under the same license as the Gollek SDK.
