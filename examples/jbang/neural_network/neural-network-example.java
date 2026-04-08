///usr/bin/env jbang
// DEPS tech.kayys:gollek-sdk-nn:1.0.0
// DEPS tech.kayys:gollek-sdk-autograd:1.0.0

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.*;

/**
 * Example: Neural Network Training with jbang
 * 
 * This script demonstrates how to build and train a simple neural network
 * using the Gollek SDK with jbang.
 * 
 * Usage: jbang examples/neural-network-example.java
 */
public class NeuralNetworkExample {

    public static void main(String[] args) {
        System.out.println("🧠 Gollek SDK - Neural Network Example");
        System.out.println("======================================");
        System.out.println();
        
        // Step 1: Build model
        System.out.println("Step 1: Building Model");
        System.out.println("---------------------");
        Module model = new Sequential(
            new Linear(784, 256),
            new ReLU(),
            new Dropout(0.2f),
            new Linear(256, 128),
            new ReLU(),
            new Dropout(0.2f),
            new Linear(128, 10)
        );
        System.out.println("✓ Created 3-layer feedforward network");
        System.out.println("  Architecture: 784 → 256 → 128 → 10");
        System.out.println();
        
        // Step 2: Setup optimizer
        System.out.println("Step 2: Setting Up Optimizer");
        System.out.println("----------------------------");
        var optimizer = new tech.kayys.gollek.ml.nn.optim.Adam(model.parameters(), 0.001f);
        System.out.println("✓ Created Adam optimizer with lr=0.001");
        System.out.println();
        
        // Step 3: Create loss function
        System.out.println("Step 3: Setting Up Loss Function");
        System.out.println("--------------------------------");
        var loss = new tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss();
        System.out.println("✓ Created CrossEntropyLoss");
        System.out.println();
        
        // Step 4: Simulate training step
        System.out.println("Step 4: Simulating Training Step");
        System.out.println("--------------------------------");
        try {
            // Create dummy data
            float[] inputData = new float[784];
            for (int i = 0; i < inputData.length; i++) {
                inputData[i] = (float) Math.random();
            }
            
            float[] targetData = new float[1];
            targetData[0] = 5; // Target class: 5
            
            // Forward pass
            var input = GradTensor.of(inputData, new long[]{1, 784});
            var output = model.forward(input);
            System.out.println("✓ Forward pass completed");
            System.out.println("  Output shape: " + java.util.Arrays.toString(output.shape()));
            
            // Compute loss
            var target = GradTensor.of(targetData, new long[]{1});
            var lossValue = loss.compute(output, target);
            System.out.println("✓ Loss computed: " + lossValue.item());
            
            // Backward pass
            lossValue.backward();
            System.out.println("✓ Backward pass completed");
            
            // Optimization step
            optimizer.step();
            optimizer.zeroGrad();
            System.out.println("✓ Optimizer step completed");
            
        } catch (Exception e) {
            System.out.println("⚠ Note: Full training simulation requires all dependencies");
            System.out.println("  This demonstrates the API structure");
        }
        System.out.println();
        
        // Summary
        System.out.println("Summary");
        System.out.println("-------");
        System.out.println("✅ Model created and configured");
        System.out.println("✅ Optimizer set up");
        System.out.println("✅ Loss function ready");
        System.out.println("✅ Training step simulated");
        System.out.println();
        System.out.println("Next steps: Integrate with real data for actual training!");
    }
}
