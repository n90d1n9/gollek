//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import java.nio.file.Paths;

/**
 * Demonstrates Model Persistence: Saving and Loading model weights.
 */
public class model_persistence {
    public static void main(String[] args) throws Exception {
        String modelPath = "xor_model.bin";
        
        System.out.println("Creating a new model...");
        Sequential model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );

        // In a real scenario, you would train the model here...
        
        System.out.println("Saving model to " + modelPath + "...");
        model.save(Paths.get(modelPath));
        System.out.println("Model saved successfully!");

        System.out.println("\nCreating a fresh model instance...");
        Sequential newModel = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );

        System.out.println("Loading weights into the fresh model...");
        newModel.load(Paths.get(modelPath));
        System.out.println("Model weights loaded successfully!");
        
        // Clean up
        Paths.get(modelPath).toFile().deleteOnExit();
    }
}
