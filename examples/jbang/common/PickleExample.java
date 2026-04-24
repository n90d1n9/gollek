package tech.kayys.gollek.ml.example;

import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.serialize.*;

/**
 * Complete example demonstrating pickle serialization.
 */
public class PickleExample {
    
    public static void main(String[] args) {
        try {
            System.out.println("╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║           Pickle Serialization Example                       ║");
            System.out.println("║      Save and Load Gollek Models with Python Compatibility   ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝\n");
            
            // 1. Train a model
            System.out.println("1. Training Random Forest model...");
            RandomForestClassifier rf = new RandomForestClassifier();
            
            // Load sample data
            float[][] X = generateSampleData(1000);
            int[] y = generateSampleLabels(1000);
            
            rf.fit(X, y);
            double accuracy = rf.score(X, y);
            System.out.printf("   Training accuracy: %.4f\n", accuracy);
            
            // 2. Save as pickle
            System.out.println("\n2. Saving model as pickle...");
            PickleSerializer.toPickle(rf, "random_forest_model.pkl");
            System.out.println("   Saved: random_forest_model.pkl");
            
            // 3. Save as compressed pickle
            System.out.println("\n3. Saving compressed pickle...");
            PickleSerializer.toPickleGZ(rf, "random_forest_model.pkl.gz");
            System.out.println("   Saved: random_forest_model.pkl.gz");
            
            // 4. Load from pickle
            System.out.println("\n4. Loading model from pickle...");
            RandomForestClassifier loadedRf = PickleSerializer.fromPickle("random_forest_model.pkl");
            double loadedAccuracy = loadedRf.score(X, y);
            System.out.printf("   Loaded model accuracy: %.4f\n", loadedAccuracy);
            
            // 5. Export for Python
            System.out.println("\n5. Exporting for Python...");
            PythonBridge.exportToPython(rf, "random_forest_model");
            System.out.println("   Generated: random_forest_model.pkl");
            System.out.println("   Generated: random_forest_model_loader.py");
            System.out.println("   Generated: random_forest_model_metadata.json");
            
            // 6. Save as NumPy format
            System.out.println("\n6. Saving data as NumPy format...");
            NumPyBridge.saveAsNumpy(X, "data.npy");
            System.out.println("   Saved: data.npy");
            
            // 7. Load from NumPy
            System.out.println("\n7. Loading from NumPy format...");
            float[][] loadedX = NumPyBridge.loadFromNumpy("data.npy");
            System.out.printf("   Loaded data shape: %d x %d\n", loadedX.length, loadedX[0].length);
            
            // 8. Verify round-trip
            System.out.println("\n8. Verifying round-trip serialization...");
            boolean match = Arrays.deepEquals(X, loadedX);
            System.out.printf("   Data integrity: %s\n", match ? "PASSED" : "FAILED");
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║                    ✓ SUCCESS!                                ║");
            System.out.println("║    Models can be saved and loaded across Java/Python        ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static float[][] generateSampleData(int nSamples) {
        float[][] X = new float[nSamples][4];
        Random rng = new Random(42);
        
        for (int i = 0; i < nSamples; i++) {
            int cluster = i % 3;
            double x1 = rng.nextGaussian() + (cluster == 0 ? -3 : cluster == 1 ? 0 : 3);
            double x2 = rng.nextGaussian() + (cluster == 0 ? -3 : cluster == 1 ? 3 : 0);
            
            X[i][0] = (float) (x1 + rng.nextGaussian() * 0.5);
            X[i][1] = (float) (x2 + rng.nextGaussian() * 0.5);
            X[i][2] = (float) rng.nextGaussian();
            X[i][3] = (float) rng.nextGaussian();
        }
        
        return X;
    }
    
    private static int[] generateSampleLabels(int nSamples) {
        int[] y = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            y[i] = i % 3;
        }
        return y;
    }
}