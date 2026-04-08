# Gollek SDK Integration Examples - Quick Start Guide

This directory contains comprehensive examples demonstrating how to integrate the Gollek SDK with popular Java ML/NLP libraries.

## 📁 Available Examples

| Library | File | Use Case | Complexity |
|---------|------|----------|------------|
| **Deeplearning4j** | `deeplearning4j_integration.java` | Image preprocessing, transfer learning, ensemble | ⭐⭐⭐ |
| **Stanford NLP** | `stanford_nlp_integration.java` | Text preprocessing, POS tagging, NER, sentiment | ⭐⭐⭐⭐ |
| **Apache OpenNLP** | `opennlp_integration.java` | Tokenization, sentence detection, NER | ⭐⭐ |
| **Smile ML** | `smile_ml_integration.java` | Feature selection, statistical features, ensemble | ⭐⭐⭐ |
| **Oracle Tribuo** | `tribuo_integration.java` | XGBoost, text classification, regression | ⭐⭐⭐⭐ |

## 🚀 Quick Start

### Prerequisites

```bash
# Java 21+ required
java --version

# jbang installed
jbang version
```

### Running Examples

```bash
cd gollek/sdk/integration/jbang-templates/examples/integration

# Run any example (first run downloads dependencies)
jbang deeplearning4j_integration.java
jbang stanford_nlp_integration.java
jbang opennlp_integration.java
jbang smile_ml_integration.java
jbang tribuo_integration.java
```

## 📚 Integration Patterns

### Pattern 1: External Preprocessing → Gollek Inference

**Libraries**: Deeplearning4j, Smile, OpenNLP

```java
// External library handles preprocessing
ExternalPreprocessor preprocessor = new ExternalPreprocessor();
float[] processed = preprocessor.process(rawData);

// Gollek handles inference
GradTensor input = GradTensor.of(processed, 1, processed.length);
GradTensor output = gollekModel.forward(input);
```

**Use Case**: Leverage mature preprocessing pipelines from established libraries while using Gollek for lightweight inference.

### Pattern 2: Feature Extraction → Gollek Classification

**Libraries**: Stanford NLP, OpenNLP, Tribuo

```java
// Extract linguistic features
float[] features = nlpPipeline.extractFeatures(text);

// Classify with Gollek
GradTensor input = GradTensor.of(features, 1, features.length);
Prediction pred = classifier.predict(input);
```

**Use Case**: Rich feature extraction from NLP libraries, neural classification with Gollek.

### Pattern 3: Ensemble (External + Gollek)

**Libraries**: All

```java
// Get predictions from both models
float externalScore = externalModel.predict(input);
float gollekScore = gollekModel.predict(input);

// Ensemble: weighted average
float ensembleScore = 0.5f * externalScore + 0.5f * gollekScore;
```

**Use Case**: Improve accuracy by combining strengths of different frameworks.

### Pattern 4: Feature Selection → Gollek Training

**Libraries**: Smile, Tribuo

```java
// Use external library for feature importance
int[] selectedFeatures = featureSelector.select(data, labels);

// Train Gollek on selected features
float[][] reducedData = selectFeatures(data, selectedFeatures);
gollekModel.train(reducedData, labels);
```

**Use Case**: Reduce dimensionality using established feature selection algorithms.

### Pattern 5: Tensor Conversion

**Libraries**: Deeplearning4j, Smile

```java
// DL4J INDArray → Gollek GradTensor
INDArray dl4jArray = ...;
GradTensor tensor = indArrayToGradTensor(dl4jArray);

// Gollek GradTensor → DL4J INDArray
GradTensor tensor = ...;
INDArray dl4jArray = gradTensorToIndArray(tensor);
```

**Use Case**: Seamless data exchange between frameworks.

## 📖 Detailed Examples

### Deeplearning4j Integration

**File**: `deeplearning4j_integration.java`

**Patterns Demonstrated**:
1. DL4J preprocessing → Gollek inference
2. DL4J feature extraction → Gollek classification
3. Ensemble (DL4J + Gollek)
4. Bidirectional tensor conversion

**Key Classes**:
- `DL4JPreprocessorGollekInference`
- `DL4JFeatureExtractorGollekClassifier`
- `EnsembleDL4JGollek`

**Run**:
```bash
jbang deeplearning4j_integration.java
```

**Dependencies**:
- deeplearning4j-core
- nd4j-native-platform
- gollek-sdk-nn

### Stanford NLP Integration

**File**: `stanford_nlp_integration.java`

**Patterns Demonstrated**:
1. Linguistic feature extraction
2. Text classification with POS/NER features
3. Hybrid sentiment analysis (Stanford + Gollek)
4. Multi-stage NLP pipeline

**Key Classes**:
- `LinguisticFeatureExtractor`
- `LinguisticTextClassifier`
- `HybridSentimentAnalyzer`

**Features Extracted**:
- Sentence length, avg word length
- POS tag distribution (12 tags)
- Named entity counts (7 categories)
- Stanford sentiment score
- Parse tree depth

**Run**:
```bash
jbang stanford_nlp_integration.java
```

**Dependencies**:
- stanford-corenlp (with models)
- gollek-sdk-nn

### Apache OpenNLP Integration

**File**: `opennlp_integration.java`

**Patterns Demonstrated**:
1. Sentence detection → tokenization → POS tagging
2. Named entity recognition
3. Text classification with OpenNLP features
4. NER-based document categorization

**Key Classes**:
- `OpenNLPPipeline`
- `OpenNLPTextClassifier`
- `NERBasedClassifier`

**Features Extracted**:
- Sentence/token counts
- POS ratios (noun, verb, adjective, adverb)
- Named entity distribution
- Vocabulary richness

**Run**:
```bash
jbang opennlp_integration.java
```

**Dependencies**:
- opennlp-tools
- gollek-sdk-nn

**Note**: Downloads OpenNLP models on first run (cached in temp directory).

### Smile ML Integration

**File**: `smile_ml_integration.java`

**Patterns Demonstrated**:
1. Smile preprocessing → Gollek inference
2. Smile feature selection → Gollek classification
3. Ensemble (Smile Random Forest + Gollek NN)
4. Statistical feature extraction

**Key Classes**:
- `SmilePreprocessorGollekModel`
- `SmileFeatureSelectionGollekClassifier`
- `EnsembleSmileGollek`
- `StatisticalFeatureExtractor`

**Statistical Features**:
- Mean, variance, standard deviation
- Min, max, skewness, kurtosis

**Run**:
```bash
jbang smile_ml_integration.java
```

**Dependencies**:
- smile-core, smile-data, smile-math
- gollek-sdk-nn

### Oracle Tribuo Integration

**File**: `tribuo_integration.java`

**Patterns Demonstrated**:
1. Tribuo feature pipeline → Gollek classification
2. Ensemble (Tribuo XGBoost + Gollek NN)
3. Tribuo text classification → Gollek
4. Tribuo regression + Gollek refinement

**Key Classes**:
- `TribuoFeaturePipelineGollekClassifier`
- `EnsembleTribuoXGBoostGollek`
- `TribuoTextPipelineGollekClassifier`
- `TribuoRegressionGollekRefinement`

**Run**:
```bash
jbang tribuo_integration.java
```

**Dependencies**:
- tribuo-all
- gollek-sdk-nn

## 🔧 Common Utilities

### Tensor Conversion (Deeplearning4j)

```java
// DL4J → Gollek
INDArray indArray = Nd4j.create(data, shape);
GradTensor tensor = indArrayToGradTensor(indArray);

// Gollek → DL4J
GradTensor tensor = GradTensor.of(data, shape);
INDArray indArray = gradTensorToIndArray(tensor);
```

### Dataset Generation

All examples include demo dataset generators:

```java
// Classification data
MutableDataset<Label> dataset = DemoDataset.generateClassificationData(
    numSamples, numFeatures, random
);

// Regression data
MutableDataset<Regressor> dataset = DemoDataset.generateRegressionData(
    numSamples, numFeatures, random
);

// Text data
List<String> texts = DemoDataset.getSampleTexts();
List<String> labels = DemoDataset.getSampleLabels();
```

## 📊 Performance Tips

### First Run Performance

First execution downloads dependencies (1-2 minutes):
```bash
# Pre-download dependencies
jbang cache deps deeplearning4j_integration.java

# Subsequent runs are fast (1-2 seconds)
jbang deeplearning4j_integration.java
```

### Memory Configuration

Some libraries require significant memory:
```bash
# Increase heap size
JAVA_OPTS="-Xmx4g" jbang stanford_nlp_integration.java
```

### Model Caching

Cache loaded models for reuse:
```java
// Static model instance
static final Model CACHED_MODEL = loadModel();

// Pre-warm
static {
    CACHED_MODEL.forward(warmupInput);
}
```

## 🐛 Troubleshooting

### Dependency Download Failures

```bash
# Clear jbang cache
jbang cache clear

# Retry
jbang deeplearning4j_integration.java
```

### Native Library Errors

Ensure `--enable-native-access=ALL-UNNAMED` is in JAVA_OPTIONS (already included in examples).

### Memory Issues

```bash
# Increase memory
export JAVA_OPTS="-Xmx4g -Xms2g"
jbang smile_ml_integration.java
```

### Model Download Failures (OpenNLP)

```bash
# Manually download models
mkdir -p /tmp/opennlp-models
cd /tmp/opennlp-models
wget https://opennlp.sourceforge.net/models-1.5/en-sent.bin
# ... other models

# Re-run example
jbang opennlp_integration.java
```

## 📚 Additional Resources

### Library Documentation

- [Deeplearning4j](https://deeplearning4j.konduit.ai/)
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- [Apache OpenNLP](https://opennlp.apache.org/)
- [Smile ML](https://haifengl.github.io/)
- [Oracle Tribuo](https://tribuo.org/)

### Gollek SDK

- [Gollek SDK README](../../../README.md)
- [JBANG Setup Guide](../JBANG_SETUP.md)
- [Examples Directory](../examples/)

### jbang

- [jbang Documentation](https://jbang.dev/documentation)
- [jbang Templates](https://github.com/jbangdev/jbang-templates)

## 🤝 Contributing

When adding new integration examples:

1. Follow naming convention: `{library}_integration.java`
2. Include comprehensive comments
3. Demonstrate 2-3 use cases
4. Include error handling
5. Add to README.md
6. Test with `jbang {file}.java`

## 📄 License

These examples are provided under the same license as the Gollek SDK.

---

**Happy integrating! 🚀**
