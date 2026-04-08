# Jupyter & jbang Examples

Practical examples using Gollek SDK with Jupyter notebooks and jbang scripts.

## Jupyter Notebooks

### 1. Getting Started Notebook

**File**: `notebooks/01-getting-started.ipynb`

Topics:
- Importing Gollek SDK
- Creating simple layers
- Building sequential models
- Running forward passes
- Understanding tensor shapes

### 2. Neural Network Training

**File**: `notebooks/02-neural-network-training.ipynb`

Topics:
- Building a classifier
- Setting up optimizers (SGD, Adam)
- Computing loss (CrossEntropyLoss)
- Training loop implementation
- Monitoring loss and accuracy

### 3. Advanced Activations

**File**: `notebooks/03-advanced-activations.ipynb`

Topics:
- Comparing activation functions (ReLU, ELU, Mish, etc.)
- When to use each activation
- Visualizing activation curves
- Impact on training dynamics

### 4. Transformers and Attention

**File**: `notebooks/04-transformers.ipynb`

Topics:
- Multi-head attention mechanism
- Causal masking for autoregressive models
- Transformer encoder/decoder
- Positional embeddings
- Sequence-to-sequence models

### 5. Batch Normalization

**File**: `notebooks/05-batch-normalization.ipynb`

Topics:
- Training vs evaluation modes
- Running statistics
- Impact on learning rates
- Combining with other layers

### 6. LLM Integration

**File**: `notebooks/06-llm-integration.ipynb`

Topics:
- Using langchain4j with Gollek
- Building LLM-powered applications
- Fine-tuning strategies
- Integration patterns

## jbang Scripts

### 1. Simple Example

**File**: `sdk/integration/jbang-templates/gollek-template.java`

Run:
```bash
jbang sdk/integration/jbang-templates/gollek-template.java
```

Output:
```
🚀 Gollek SDK jbang Template
============================
...
```

### 2. Neural Network Training

**File**: `examples/neural-network-example.java`

Run:
```bash
jbang examples/neural-network-example.java
```

Features:
- Builds sequential model
- Creates optimizer
- Simulates training step
- Shows API usage

### 3. Batch Data Processing

**File**: `examples/batch-processor.java`

Run:
```bash
jbang examples/batch-processor.java /path/to/data
```

Features:
- Reads CSV files
- Processes in batches
- Handles errors gracefully

### 4. Model Export

**File**: `examples/model-export.java`

Run:
```bash
jbang examples/model-export.java --input model.dat --output model.json
```

Features:
- Saves trained models
- Exports configuration
- Version tracking

### 5. Multimodal AI (Real-World)
 
 **Directory**: `examples/multimodal/`
 
 These examples demonstrate the unified Vision, Audio, and Video capabilities of the Gollek SDK using the new fluent API.
 
 #### A. Vision Storyteller
 **File**: `examples/multimodal/vision_storyteller.java`
 ```bash
 jbang examples/multimodal/vision_storyteller.java sample.jpg
 ```
 *Generates a creative story from an image.*
 
 #### B. Smart Transcriber
 **File**: `examples/multimodal/smart_transcriber.java`
 ```bash
 jbang examples/multimodal/smart_transcriber.java meeting.wav
 ```
 *Transcribes audio and extracts key highlights.*
 
 #### C. Video Analyst
 **File**: `examples/multimodal/video_analyst.java`
 ```bash
 jbang examples/multimodal/video_analyst.java clip.mp4
 ```
 *Second-by-second temporal scene analysis.*
 
 #### D. Omni Assistant
 **File**: `examples/multimodal/omni_assistant.java`
 ```bash
 jbang examples/multimodal/omni_assistant.java house.jpg note.wav "How do these relate?"
 ```
 *Mixed-modality reasoning across text, image, and audio.*
 
 ## Usage Patterns

### Pattern 1: Interactive Exploration (Jupyter)

Perfect for:
- Experimenting with architectures
- Visualizing gradients
- Learning deep learning concepts
- Prototyping solutions

**Workflow**:
1. Start Jupyter
2. Create cells
3. Import libraries
4. Build and train
5. Visualize results
6. Refine and repeat

### Pattern 2: Batch Processing (jbang)

Perfect for:
- Data preprocessing
- Model evaluation
- Automated training
- Production pipelines

**Workflow**:
1. Write jbang script
2. Add dependencies
3. Implement logic
4. Run from command line
5. Integrate into workflows

### Pattern 3: Development (Both)

Perfect for:
- Prototyping in Jupyter
- Exporting to jbang
- Testing in production
- Version control

**Workflow**:
1. Develop in Jupyter
2. Export working code
3. Convert to jbang script
4. Add CLI arguments
5. Test and deploy

## Tips for Success

### Jupyter Tips

1. **Use Markdown cells** for explanations
2. **Keep cells small** for reusability
3. **Clear output** between experiments
4. **Use print statements** for debugging
5. **Document assumptions** clearly

### jbang Tips

1. **Start with template** to learn structure
2. **Use CLI arguments** for flexibility
3. **Add error handling** for robustness
4. **Cache large objects** across runs
5. **Keep dependencies minimal**

### General Tips

1. **Test locally** before sharing
2. **Use version control** for reproducibility
3. **Document examples** thoroughly
4. **Provide expected output** for validation
5. **Include troubleshooting** guidance

## Creating Your Own

### From Jupyter to jbang

1. Write and test in Jupyter
2. Copy working cells to jbang template
3. Add argument parsing
4. Add error handling
5. Test from command line

```java
// Convert this Jupyter cell:
Module model = new Sequential(...);

// Into this jbang script:
///usr/bin/env jbang
// DEPS tech.kayys:gollek-sdk-nn:1.0.0

public class MyModel {
    public static void main(String[] args) {
        Module model = new Sequential(...);
        // Use model...
    }
}
```

### From jbang to Jupyter

1. Take jbang script
2. Extract class definition
3. Paste into Jupyter cell
4. Remove main() wrapper
5. Execute cells sequentially

## Common Patterns

### Pattern: Train and Evaluate

```java
// Build model
Module model = new Sequential(...);

// Train
for (int epoch = 0; epoch < epochs; epoch++) {
    // Forward pass
    // Backward pass
    // Optimizer step
}

// Evaluate
Accuracy metric = new Accuracy();
// Evaluate on test set
```

### Pattern: Save and Load

```java
// Save model weights
// model.save("model.dat");

// Load model weights
// model.load("model.dat");
```

### Pattern: Hyperparameter Search

```java
// Try different parameters
for (float lr : lrs) {
    for (int hidden : hiddenSizes) {
        // Build model with parameters
        // Train and evaluate
        // Track best
    }
}
```

### Pattern: Ensemble Models

```java
// Train multiple models
Module[] models = new Module[numModels];
for (int i = 0; i < numModels; i++) {
    models[i] = buildModel();
    // Train models[i]
}

// Ensemble prediction
var predictions = new float[numClasses];
for (var model : models) {
    var pred = model.forward(input);
    // Aggregate
}
```

## Learning Path

### Beginner

1. Read JUPYTER_SETUP.md
2. Run 01-getting-started.ipynb
3. Try simple jbang script
4. Modify examples

### Intermediate

1. Work through neural network notebook
2. Write custom jbang script
3. Combine Jupyter + jbang workflow
4. Experiment with architectures

### Advanced

1. Study transformer notebook
2. Build complex pipelines
3. Integrate with production systems
4. Optimize performance

## Next Steps

1. **Install** Jupyter/jbang (see setup guides)
2. **Clone/download** example notebooks and scripts
3. **Follow** one of the examples
4. **Modify** to your use case
5. **Share** your creations!

## Resources

- **Jupyter Setup**: [JUPYTER_SETUP.md](../jupyter-kernel/JUPYTER_SETUP.md)
- **jbang Setup**: [JBANG_SETUP.md](../jbang-templates/JBANG_SETUP.md)
- **API Reference**: [API_REFERENCE.md](../API_REFERENCE.md)
- **Gollek SDK**: [README.md](../README.md)

---

**Happy Learning! 🚀**
