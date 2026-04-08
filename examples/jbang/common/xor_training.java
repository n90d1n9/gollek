//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.*;
import tech.kayys.gollek.ml.nn.optim.*;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * A complete, runnable example of training a Neural Network to solve the XOR problem.
 * XOR is a classic non-linear problem that requires at least one hidden layer.
 */
public class xor_training {
    public static void main(String[] args) {
        System.out.println("=== Training Gollek NN on XOR Problem ===");

        // 1. Prepare Data (XOR Input: [0,0], [0,1], [1,0], [1,1])
        float[] xData = {0, 0, 0, 1, 1, 0, 1, 1};
        float[] yData = {0, 1, 1, 0};
        
        GradTensor x = GradTensor.of(xData, 4, 2);
        GradTensor y = GradTensor.of(yData, 4, 1);

        // 2. Define Model (2 inputs -> 8 hidden -> 1 output)
        Sequential model = new Sequential(
            new Linear(2, 8),
            new ReLU(),
            new Linear(8, 1)
            // Note: Sigmoid is handled internally by BCEWithLogitsLoss
        );

        // 3. Loss and Optimizer
        BCEWithLogitsLoss criterion = new BCEWithLogitsLoss(); 
        SGD optimizer = new SGD(model.parameters(), 0.1f);

        // 4. Training Loop
        System.out.println("Starting training...");
        for (int epoch = 1; epoch <= 500; epoch++) {
            // Forward pass
            GradTensor output = model.forward(x);
            GradTensor loss = criterion.compute(output, y);

            // Backward pass
            model.zeroGrad();
            loss.backward();

            // Weight update
            optimizer.step();

            if (epoch % 50 == 0) {
                System.out.printf("Epoch [%d/500], Loss: %.4f%n", epoch, loss.item());
            }
        }

        // 5. Evaluation
        model.eval();
        GradTensor finalOutput = model.forward(x);
        System.out.println("\nFinal Predictions:");
        for (int i = 0; i < 4; i++) {
            float pred = finalOutput.data()[i];
            System.out.printf("Input: [%.0f, %.0f] -> Target: %.0f -> Predicted: %.4f%n", 
                xData[i*2], xData[i*2+1], yData[i], pred);
        }
    }
}
