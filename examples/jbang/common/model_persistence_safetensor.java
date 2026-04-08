//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.optim.SGD;
import tech.kayys.gollek.ml.autograd.GradTensor;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * End-to-End Example: Training, Saving, Loading, and Evaluating with Safetensors.
 * 
 * This script demonstrates the full lifecycle of a Gollek model:
 * 1. Define a non-linear XOR model.
 * 2. Train it for a few epochs.
 * 3. Save the trained weights to the universal Safetensor format.
 * 4. Create a fresh, random instance of the same model architecture.
 * 5. Load the trained weights via loadSafetensors().
 * 6. Verify that the reloaded model's predictions match the original.
 */
public class model_persistence_safetensor {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Gollek End-to-End Persistence (Safetensor) ===");

        // --- 1. SETUP DATA & MODEL ---
        GradTensor x = GradTensor.of(new float[]{0,0, 0,1, 1,0, 1,1}, 4, 2);
        GradTensor y = GradTensor.of(new float[]{0, 1, 1, 0}, 4, 1);

        Sequential model = new Sequential(
            new Linear(2, 8),
            new ReLU(),
            new Linear(8, 1)
        );

        // --- 2. QUICK TRAINING ---
        System.out.println("\n[1/4] Training the original model...");
        long startTrain = System.nanoTime();
        BCEWithLogitsLoss criterion = new BCEWithLogitsLoss();
        SGD optimizer = new SGD(model.parameters(), 0.5f);

        for (int i = 1; i <= 200; i++) {
            GradTensor output = model.forward(x);
            GradTensor loss = criterion.compute(output, y);
            model.zeroGrad();
            loss.backward();
            optimizer.step();
            if (i % 50 == 0) System.out.printf("Epoch %d, Loss: %.4f\n", i, loss.item());
        }
        long trainTime = (System.nanoTime() - startTrain) / 1_000_000;
        System.out.printf("Training complete in %d ms.\n", trainTime);

        // Capture original prediction for later comparison
        float originalPred = model.forward(GradTensor.of(new float[]{1, 0}, 1, 2)).item();
        System.out.printf("Original model prediction for [1,0]: %.4f\n", originalPred);

        // --- 3. SAVE TO SAFETENSOR ---
        String filename = "trained_xor.safetensors";
        System.out.println("\n[2/4] Saving model to " + filename + "...");
        long startSave = System.nanoTime();
        model.saveSafetensors(filename);
        long saveTime = (System.nanoTime() - startSave) / 1_000_000;
        System.out.printf("Save complete in %d ms.\n", saveTime);

        // --- 4. RELOAD INTO FRESH MODEL ---
        System.out.println("\n[3/4] Creating a fresh model and loading weights...");
        Sequential freshModel = new Sequential(
            new Linear(2, 8),
            new ReLU(),
            new Linear(8, 1)
        );

        float randomPred = freshModel.forward(GradTensor.of(new float[]{1, 0}, 1, 2)).item();
        System.out.printf("Fresh (random) model prediction for [1,0]: %.4f\n", randomPred);

        System.out.println("Loading weights from Safetensor file...");
        long startLoad = System.nanoTime();
        freshModel.loadSafetensors(filename);
        long loadTime = (System.nanoTime() - startLoad) / 1_000_000;
        System.out.printf("Load complete in %d ms.\n", loadTime);

        // --- 5. EVALUATE & VERIFY ---
        System.out.println("\n[4/4] Evaluating reloaded model...");
        float reloadedPred = freshModel.forward(GradTensor.of(new float[]{1, 0}, 1, 2)).item();
        System.out.printf("Reloaded model prediction for [1,0]: %.4f\n", reloadedPred);

        if (Math.abs(originalPred - reloadedPred) < 1e-6) {
            System.out.println("\n✅ SUCCESS: Reloaded model matches original model perfectly!");
        } else {
            System.out.println("\n❌ FAILURE: Predictive discrepancy detected.");
        }
        
        // Clean up
        java.nio.file.Files.deleteIfExists(Path.of(filename));
    }
}
