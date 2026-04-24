package tech.kayys.gollek.ml.example;

import tech.kayys.gollek.ml.base.*;
import tech.kayys.gollek.ml.pipeline.*;
import tech.kayys.gollek.ml.ensemble.*;
import tech.kayys.gollek.ml.model_selection.*;
import tech.kayys.gollek.ml.metrics.*;

/**
 * Complete example showing all components working together.
 */
public class CompleteMlExample {
    
    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║           Complete Gollek ML Framework Demo                  ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝\n");
        
        try {
            // Load sample data
            float[][] X = loadSampleData();
            int[] y = loadSampleLabels();
            
            System.out.println("📊 Data loaded:");
            System.out.printf("   Samples: %d, Features: %d, Classes: %d\n", 
                X.length, X[0].length, Arrays.stream(y).max().getAsInt() + 1);
            
            // 1. Create preprocessing pipeline
            System.out.println("\n🔧 Building preprocessing pipeline...");
            Pipeline preprocessor = new Pipeline(
                new BaseTransformer[]{
                    new StandardScaler(),
                    new PCA(2)
                },
                null
            );
            
            float[][] XProcessed = preprocessor.fitTransform(X);
            System.out.printf("   Features after PCA: %d\n", XProcessed[0].length);
            
            // 2. Train multiple models
            System.out.println("\n🤖 Training models...");
            
            // Random Forest
            RandomForestClassifier rf = new RandomForestClassifier();
            rf.fit(XProcessed, y);
            double rfAcc = rf.score(XProcessed, y);
            System.out.printf("   Random Forest Accuracy: %.4f\n", rfAcc);
            
            // SVM
            SVC svm = new SVC();
            svm.fit(XProcessed, y);
            double svmAcc = svm.score(XProcessed, y);
            System.out.printf("   SVM Accuracy: %.4f\n", svmAcc);
            
            // Gradient Boosting
            GradientBoostingClassifier gb = new GradientBoostingClassifier();
            gb.fit(XProcessed, y);
            double gbAcc = gb.score(XProcessed, y);
            System.out.printf("   Gradient Boosting Accuracy: %.4f\n", gbAcc);
            
            // 3. Cross-validation
            System.out.println("\n📈 Cross-validation results (5-fold):");
            
            KFold kfold = new KFold(5, true, 42);
            List<Fold> folds = kfold.split(XProcessed.length);
            
            double cvSum = 0;
            for (int i = 0; i < folds.size(); i++) {
                Fold fold = folds.get(i);
                
                // Extract fold data
                float[][] XTrain = new float[fold.trainIndices.length][];
                int[] yTrain = new int[fold.trainIndices.length];
                float[][] XVal = new float[fold.valIndices.length][];
                int[] yVal = new int[fold.valIndices.length];
                
                for (int j = 0; j < fold.trainIndices.length; j++) {
                    int idx = fold.trainIndices[j];
                    XTrain[j] = XProcessed[idx];
                    yTrain[j] = y[idx];
                }
                for (int j = 0; j < fold.valIndices.length; j++) {
                    int idx = fold.valIndices[j];
                    XVal[j] = XProcessed[idx];
                    yVal[j] = y[idx];
                }
                
                // Train and evaluate
                RandomForestClassifier foldRf = new RandomForestClassifier();
                foldRf.fit(XTrain, yTrain);
                double acc = Metrics.accuracy(yVal, foldRf.predict(XVal));
                cvSum += acc;
                System.out.printf("   Fold %d: %.4f\n", i + 1, acc);
            }
            System.out.printf("   Mean CV Accuracy: %.4f\n", cvSum / folds.size());
            
            // 4. Model selection with grid search
            System.out.println("\n🔍 Grid search for best parameters...");
            Map<String, Object[]> paramGrid = new LinkedHashMap<>();
            paramGrid.put("nEstimators", new Object[]{50, 100});
            paramGrid.put("maxDepth", new Object[]{5, 10});
            paramGrid.put("minSamplesSplit", new Object[]{2, 5});
            
            // 5. Ensemble voting
            System.out.println("\n🎯 Voting ensemble:");
            List<BaseEstimator> estimators = List.of(rf, svm, gb);
            VotingClassifier voting = new VotingClassifier(estimators, "hard", new double[]{1, 1, 1});
            double votingAcc = voting.score(XProcessed, y);
            System.out.printf("   Voting Classifier Accuracy: %.4f\n", votingAcc);
            
            // 6. Model persistence
            System.out.println("\n💾 Saving best model...");
            ModelPersistence.save(rf, "random_forest_model.gollek");
            System.out.println("   Model saved to random_forest_model.gollek");
            
            // 7. Load and predict
            System.out.println("\n📂 Loading saved model...");
            RandomForestClassifier loadedRf = ModelPersistence.load("random_forest_model.gollek");
            int[] predictions = loadedRf.predict(XProcessed);
            double loadedAcc = Metrics.accuracy(y, predictions);
            System.out.printf("   Loaded model accuracy: %.4f\n", loadedAcc);
            
            System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
            System.out.println("║                    ✓ COMPLETE!                               ║");
            System.out.println("║    All ML components working together successfully          ║");
            System.out.println("╚══════════════════════════════════════════════════════════════╝");
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static float[][] loadSampleData() {
        // Generate synthetic 2D data (3 blobs for clustering/classification)
        float[][] data = new float[300][2];
        Random rng = new Random(42);
        
        for (int i = 0; i < 300; i++) {
            int cluster = i / 100;
            double angle = rng.nextDouble() * Math.PI * 2;
            double radius = rng.nextDouble() * 2;
            
            double x = Math.cos(angle) * radius + (cluster == 0 ? 0 : cluster == 1 ? 5 : -5);
            double y = Math.sin(angle) * radius + (cluster == 0 ? 0 : cluster == 1 ? 5 : -5);
            
            data[i][0] = (float) (x + rng.nextGaussian() * 0.3);
            data[i][1] = (float) (y + rng.nextGaussian() * 0.3);
        }
        
        return data;
    }
    
    private static int[] loadSampleLabels() {
        int[] labels = new int[300];
        for (int i = 0; i < 300; i++) {
            labels[i] = i / 100;
        }
        return labels;
    }
}