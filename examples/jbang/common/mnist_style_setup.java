//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.*;
import tech.kayys.gollek.ml.nn.optim.*;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Simulates a complex Digit Classification setup (like MNIST).
 * Demonstrates deeper networks, parameter counting, and training/eval modes.
 */
public class mnist_style_setup {
    public static void main(String[] args) {
        int inputSize = 28 * 28; // 784
        int hiddenSize = 256;
        int numClasses = 10;

        // 1. Advanced Model Architecture
        Sequential model = new Sequential(
            new Linear(inputSize, hiddenSize),
            new ReLU(),
            // new Dropout(0.2), // If supported
            new Linear(hiddenSize, hiddenSize / 2),
            new ReLU(),
            new Linear(hiddenSize / 2, numClasses)
        );

        System.out.println("--- Model Summary ---");
        System.out.println(model);
        System.out.println("Total Parameters: " + model.parameterCountFormatted());

        // 2. Training Components
        CrossEntropyLoss criterion = new CrossEntropyLoss();
        Adam optimizer = new Adam(model.parameters(), 0.001f);

        // 3. Batch simulation
        int batchSize = 32;
        GradTensor fakeImages = GradTensor.randn(batchSize, inputSize);
        GradTensor fakeLabels = GradTensor.zeros(batchSize, numClasses); // One-hot labels

        System.out.println("\n--- Step Execution ---");
        
        // Train mode
        model.train();
        GradTensor output = model.forward(fakeImages);
        GradTensor loss = criterion.compute(output, fakeLabels);
        
        System.out.println("Forward pass batch output shape: " + java.util.Arrays.toString(output.shape()));
        System.out.println("Current batch loss: " + loss.item());

        // Eval mode
        model.eval();
        GradTensor valOutput = model.forward(fakeImages);
        System.out.println("Evaluation mode predictions complete.");
    }
}
