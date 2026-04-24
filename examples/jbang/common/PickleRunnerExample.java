package tech.kayys.gollek.ml.example;

import tech.kayys.gollek.ml.pickle.*;
import tech.kayys.gollek.ml.base.*;
import java.util.*;

/**
 * Complete example demonstrating loading and running pickle models in Java.
 */
public class PickleRunnerExample {
    
    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║           Pickle Model Runner - Complete Example            ║");
        System.out.println("║      Load scikit-learn/PyTorch models in Java               ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝\n");
        
        try {
            // ============================================================
            // 1. Load Pickle Model
            // ============================================================
            System.out.println("1. Loading pickle model...");
            
            // Create pickle from Python first:
            // python -c "
            // import pickle
            // from sklearn.ensemble import RandomForestClassifier
            // import numpy as np
            // X = np.random.randn(100, 4)
            // y = np.random.randint(0, 3, 100)
            // model = RandomForestClassifier(n_estimators=10)
            // model.fit(X, y)
            // with open('model.pkl', 'wb') as f:
            //     pickle.dump(model, f)
            // "
            
            PickleModelRunner runner = new PickleModelRunner("model.pkl");
            runner.load();
            
            // Display metadata
            System.out.println("\n   Model Metadata:");
            for (var entry : runner.getMetadata().entrySet()) {
                System.out.printf("     %s: %s\n", entry.getKey(), entry.getValue());
            }
            
            // ============================================================
            // 2. Run Inference
            // ============================================================
            System.out.println("\n2. Running inference...");
            
            // Generate test samples
            Random rng = new Random(42);
            float[][] testSamples = new float[10][4];
            for (int i = 0; i < testSamples.length; i++) {
                for (int j = 0; j < 4; j++) {
                    testSamples[i][j] = (float) rng.nextGaussian();
                }
            }
            
            // Predict
            long startTime = System.currentTimeMillis();
            int[] predictions = runner.predictBatch(testSamples);
            long inferenceTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("   Inference completed in %d ms\n", inferenceTime);
            System.out.println("   Predictions: " + Arrays.toString(predictions));
            
            // ============================================================
            // 3. Async Inference
            // ============================================================
            System.out.println("\n3. Async inference demo...");
            
            var future = runner.predictAsync(testSamples);
            future.thenAccept(results -> {
                System.out.println("   Async results: " + Arrays.toString(results));
            }).get(); // Wait for completion
            
            // ============================================================
            // 4. Probability Predictions
            // ============================================================
            System.out.println("\n4. Getting probabilities...");
            
            double[][] probs = runner.predictProba(testSamples);
            System.out.printf("   Probabilities shape: %d x %d\n", probs.length, probs[0].length);
            for (int i = 0; i < Math.min(3, probs.length); i++) {
                System.out.printf("   Sample %d: ", i);
                for (double p : probs[i]) {
                    System.out.printf("%.3f ", p);
                }
                System.out.println();
            }
            
            // ============================================================
            // 5. Performance Statistics
            // ============================================================
            System.out.println("\n5. Performance Statistics:");
            var stats = runner.getStats();
            System.out.printf("   %s\n", stats);
            System.out.printf("   Throughput: %.1f inferences/second\n", stats.throughputPerSecond());
            
            // ============================================================
            // 6. Save as Gollek Native Format
            // ============================================================
            System.out.println("\n6. Saving as Gollek native format...");
            runner.saveAsGollek("converted_model.gollek");
            System.out.println("   Saved: converted_model.gollek");
            
            // ============================================================
            // 7. Batch Processing with Different Sizes
            // ============================================================
            System.out.println("\n7. Batch size performance test:");
            
            int[] batchSizes = {1, 8, 32, 64, 128};
            for (int batchSize : batchSizes) {
                float[][] batch = new float[batchSize][4];
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < 4; j++) {
                        batch[i][j] = (float) rng.nextGaussian();
                    }
                }
                
                long batchStart = System.nanoTime();
                int[] results = runner.predictBatch(batch);
                long batchTime = System.nanoTime() - batchStart;
                
                double avgLatency = batchTime / (double) batchSize / 1_000_000.0;
                System.out.printf("   Batch size %3d: total=%.2fms, avg=%.3fms per sample\n",
                    batchSize, batchTime / 1_000_000.0, avgLatency);
            }
            
            // ============================================================
            // 8. Cleanup
            // ============================================================
            runner.close();
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║                    ✓ SUCCESS!                                ║");
            System.out.println("║    Pickle model loaded and running in pure Java              ║");
            System.out.println("║    No Python required for inference!                        ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║  Note: Create a model.pkl file first using the Python script ║");
            System.out.println("║  shown in the comments above.                                ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
        }
    }
}