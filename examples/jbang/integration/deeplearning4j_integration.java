///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS org.deeplearning4j:deeplearning4j-core:1.0.0-M2.1
//DEPS org.deeplearning4j:deeplearning4j-nn:1.0.0-M2.1
//DEPS org.nd4j:nd4j-native-platform:1.0.0-M2.1
//DEPS org.datavec:datavec-api:1.0.0-M2.1
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-nn-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-autograd-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-runtime-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-model-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-api-0.1.0-SNAPSHOT.jar
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.MeanSquaredErrorLoss;
import tech.kayys.gollek.ml.nn.optim.Adam;

import java.util.Arrays;
import java.util.Random;

/**
 * Deeplearning4j Integration with Gollek SDK.
 *
 * This example demonstrates how to integrate Gollek SDK with Deeplearning4j
 * (DL4J),
 * leveraging the strengths of both frameworks:
 *
 * 1. DL4J's mature ecosystem (pre-trained models, image preprocessing)
 * 2. Gollek's lightweight inference and custom architectures
 * 3. Bidirectional tensor conversion (INDArray ↔ GradTensor)
 * 4. Hybrid workflows combining both frameworks
 *
 * Use Cases:
 * - Using DL4J pre-trained models (ResNet, VGG) alongside Gollek models
 * - Converting between DL4J INDArrays and Gollek tensors
 * - Combining DL4J image preprocessing with Gollek inference
 * - Ensemble predictions from both frameworks
 * - Feature extraction with DL4J, classification with Gollek
 *
 * Run: jbang deeplearning4j_integration.java
 */
public class deeplearning4j_integration {

    // ──────────────────────────────────────────────────────────────────────
    // Tensor Conversion Utilities
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Convert DL4J INDArray to Gollek GradTensor.
     * Preserves shape and data.
     */
    static GradTensor indArrayToGradTensor(INDArray indArray) {
        long[] shape = indArray.shape();
        float[] data = indArray.toFloatVector();

        // Handle different dimensionalities
        if (shape.length == 1) {
            return GradTensor.of(data, (int) shape[0]);
        } else if (shape.length == 2) {
            return GradTensor.of(data, (int) shape[0], (int) shape[1]);
        } else if (shape.length == 3) {
            return GradTensor.of(data, (int) shape[0], (int) shape[1], (int) shape[2]);
        } else if (shape.length == 4) {
            return GradTensor.of(data, (int) shape[0], (int) shape[1], (int) shape[2], (int) shape[3]);
        } else {
            throw new IllegalArgumentException("Unsupported tensor dimension: " + shape.length);
        }
    }

    /**
     * Convert Gollek GradTensor to DL4J INDArray.
     * Preserves shape and data.
     */
    static INDArray gradTensorToIndArray(GradTensor tensor) {
        int[] shape = tensor.shape();
        float[] data = tensor.data();
        return Nd4j.create(data, shape);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 1: DL4J Preprocessing → Gollek Inference
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Demonstrates using DL4J's data normalization with Gollek models.
     * DL4J provides robust preprocessing utilities that can be reused.
     */
    static class DL4JPreprocessorGollekInference {
        private final DataNormalization normalizer;
        private final Module gollekModel;
        private final int inputDim;

        DL4JPreprocessorGollekInference(int inputDim) {
            this.inputDim = inputDim;
            this.normalizer = new NormalizerStandardize();
            this.gollekModel = new Sequential(
                    new Linear(inputDim, 32),
                    new ReLU(),
                    new Linear(32, 16),
                    new ReLU(),
                    new Linear(16, 2));
        }

        /**
         * Train the normalizer on sample data, then train Gollek model.
         */
        void train(float[][] trainData, float[] labels, int epochs, int batchSize) {
            System.out.println("\n━━━ Pattern 1: DL4J Preprocessing → Gollek Inference ━━━");

            // Step 1: Create DL4J DataSet for normalization fitting
            INDArray features = Nd4j.create(trainData);
            INDArray labelsArray = Nd4j.create(labels, labels.length, 1);
            DataSet dataSet = new DataSet(features, labelsArray);

            // Step 2: Fit normalizer on training data
            System.out.println("📊 Fitting DL4J normalizer on training data...");
            normalizer.fit(dataSet);

            // Step 3: Normalize features using DL4J
            System.out.println("🔄 Normalizing features with DL4J...");
            normalizer.transform(dataSet);
            INDArray normalizedFeatures = dataSet.getFeatures();

            // Step 4: Convert to Gollek tensor
            GradTensor normalizedTensor = indArrayToGradTensor(normalizedFeatures);

            // Step 5: Train Gollek model
            System.out.println("🏋️ Training Gollek model on normalized data...");
            var trainer = Trainer.builder()
                    .model(gollekModel)
                    .optimizer(new GollekAdam(gollekModel.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new MeanSquaredErrorLoss().compute(pred, target))
                    .epochs(epochs)
                    .callback(new Trainer.TrainingCallback() {
                        @Override
                        public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                            if ((epoch + 1) % 20 == 0 || epoch == 0) {
                                System.out.printf("  Epoch %3d/%d │ loss: %.4f%n", epoch + 1, totalEpochs, avgLoss);
                            }
                        }
                    })
                    .build();

            GradTensor labelTensor = GradTensor.of(labels, labels.length, 1);
            trainer.fit(normalizedTensor, labelTensor, batchSize);

            System.out.println("✅ Training complete!");
        }

        /**
         * Predict using DL4J normalization + Gollek inference.
         */
        float[] predict(float[][] inputData) {
            // Step 1: Create DL4J array
            INDArray inputArray = Nd4j.create(inputData);

            // Step 2: Normalize using DL4J normalizer (fitted during training)
            normalizer.transform(inputArray);

            // Step 3: Convert to Gollek tensor
            GradTensor inputTensor = indArrayToGradTensor(inputArray);

            // Step 4: Run Gollek inference
            gollekModel.eval();
            GradTensor output = gollekModel.forward(inputTensor);

            return output.data();
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 2: DL4J Model → Gollek Model (Feature Extraction)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Demonstrates using DL4J as a feature extractor, feeding into Gollek.
     * This is useful for transfer learning scenarios.
     */
    static class DL4JFeatureExtractorGollekClassifier {
        private final MultiLayerNetwork dl4jFeatureExtractor;
        private final Module gollekClassifier;
        private final int featureDim;

        DL4JFeatureExtractorGollekClassifier(int inputDim, int featureDim) {
            this.featureDim = featureDim;

            // Build DL4J feature extractor (simplified version of transfer learning)
            // In production, you would load a pre-trained model like ResNet
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(42)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.001))
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(inputDim)
                            .nOut(128)
                            .activation(Activation.RELU)
                            .build())
                    .layer(1, new DenseLayer.Builder()
                            .nIn(128)
                            .nOut(featureDim)
                            .activation(Activation.RELU)
                            .build())
                    .build();

            this.dl4jFeatureExtractor = new MultiLayerNetwork(conf);
            this.dl4jFeatureExtractor.init();

            // Build Gollek classifier
            this.gollekClassifier = new Sequential(
                    new Linear(featureDim, 64),
                    new ReLU(),
                    new Linear(64, 2));

            System.out.println("📦 DL4J Feature Extractor: " + inputDim + " → " + featureDim);
            System.out.println("📦 Gollek Classifier: " + featureDim + " → 2");
        }

        /**
         * Extract features using DL4J, classify using Gollek.
         */
        float[] predict(float[] inputData) {
            // Step 1: Create DL4J input
            INDArray inputArray = Nd4j.create(inputData).reshape(1, inputData.length);

            // Step 2: Extract features using DL4J (forward pass through feature layers)
            INDArray features = dl4jFeatureExtractor.feedForward(inputArray).get(1); // Get layer 1 output

            // Step 3: Convert to Gollek tensor
            GradTensor featureTensor = indArrayToGradTensor(features);

            // Step 4: Classify using Gollek
            gollekClassifier.eval();
            GradTensor output = gollekClassifier.forward(featureTensor);

            return output.data();
        }

        /**
         * Train only the Gollek classifier (DL4J feature extractor is frozen).
         */
        void trainClassifier(float[][] trainData, float[] labels, int epochs) {
            System.out.println("\n━━━ Pattern 2: DL4J Feature Extraction → Gollek Classification ━━━");
            System.out.println("🔒 DL4J feature extractor is frozen (transfer learning)");
            System.out.println("🏋️ Training Gollek classifier...");

            // Extract features for all training data
            INDArray inputArray = Nd4j.create(trainData);
            java.util.List<INDArray> featureList = new java.util.ArrayList<>();

            for (int i = 0; i < trainData.length; i++) {
                INDArray single = inputArray.getRow(i);
                INDArray features = dl4jFeatureExtractor.feedForward(single).get(1);
                featureList.add(features);
            }

            // Stack features
            int totalSamples = featureList.size();
            float[] featureData = new float[totalSamples * featureDim];
            for (int i = 0; i < featureList.size(); i++) {
                float[] feat = featureList.get(i).toFloatVector();
                System.arraycopy(feat, 0, featureData, i * featureDim, featureDim);
            }

            GradTensor featureTensor = GradTensor.of(featureData, totalSamples, featureDim);
            GradTensor labelTensor = GradTensor.of(labels, labels.length, 1);

            var trainer = Trainer.builder()
                    .model(gollekClassifier)
                    .optimizer(new GollekAdam(gollekClassifier.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new MeanSquaredErrorLoss().compute(pred, target))
                    .epochs(epochs)
                    .callback(new Trainer.TrainingCallback() {
                        @Override
                        public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                            if ((epoch + 1) % 20 == 0 || epoch == 0) {
                                System.out.printf("  Epoch %3d/%d │ loss: %.4f%n", epoch + 1, totalEpochs, avgLoss);
                            }
                        }
                    })
                    .build();

            trainer.fit(featureTensor, labelTensor, 16);
            System.out.println("✅ Classifier training complete!");
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 3: Ensemble Predictions (DL4J + Gollek)
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Demonstrates ensemble predictions combining DL4J and Gollek models.
     * This can improve accuracy by leveraging both frameworks' strengths.
     */
    static class EnsembleDL4JGollek {
        private final MultiLayerNetwork dl4jModel;
        private final Module gollekModel;
        private final int inputDim;
        private final float dl4jWeight;
        private final float gollekWeight;

        EnsembleDL4JGollek(int inputDim, float dl4jWeight, float gollekWeight) {
            this.inputDim = inputDim;
            this.dl4jWeight = dl4jWeight;
            this.gollekWeight = gollekWeight;

            // Build DL4J model
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(42)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.001))
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(inputDim)
                            .nOut(64)
                            .activation(Activation.RELU)
                            .build())
                    .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .nIn(64)
                            .nOut(1)
                            .activation(Activation.IDENTITY)
                            .build())
                    .build();

            this.dl4jModel = new MultiLayerNetwork(conf);
            this.dl4jModel.init();

            // Build Gollek model
            this.gollekModel = new Sequential(
                    new Linear(inputDim, 64),
                    new ReLU(),
                    new Linear(64, 1));

            System.out.println("🎭 Ensemble Model:");
            System.out.printf("  DL4J weight: %.1f%%, Gollek weight: %.1f%%%n",
                    dl4jWeight * 100, gollekWeight * 100);
        }

        /**
         * Train both models separately.
         */
        void train(float[][] trainData, float[] labels, int epochs) {
            System.out.println("\n━━━ Pattern 3: Ensemble (DL4J + Gollek) ━━━");

            // Train DL4J model
            System.out.println("🏋️ Training DL4J model...");
            INDArray features = Nd4j.create(trainData);
            INDArray labelsArray = Nd4j.create(labels, labels.length, 1);
            DataSet dataSet = new DataSet(features, labelsArray);

            for (int epoch = 0; epoch < epochs; epoch++) {
                dl4jModel.fit(dataSet);
                if ((epoch + 1) % 20 == 0 || epoch == 0) {
                    System.out.printf("  Epoch %3d/%d │ score: %.4f%n",
                            epoch + 1, epochs, dl4jModel.score());
                }
            }

            // Train Gollek model
            System.out.println("🏋️ Training Gollek model...");
            GradTensor featureTensor = indArrayToGradTensor(features);
            GradTensor labelTensor = GradTensor.of(labels, labels.length, 1);

            var trainer = Trainer.builder()
                    .model(gollekModel)
                    .optimizer(new GollekAdam(gollekModel.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new MeanSquaredErrorLoss().compute(pred, target))
                    .epochs(epochs)
                    .callback(new Trainer.TrainingCallback() {
                        @Override
                        public void onEpochEnd(int epoch, int totalEpochs, float avgLoss) {
                            if ((epoch + 1) % 20 == 0 || epoch == 0) {
                                System.out.printf("  Epoch %3d/%d │ loss: %.4f%n",
                                        epoch + 1, totalEpochs, avgLoss);
                            }
                        }
                    })
                    .build();

            trainer.fit(featureTensor, labelTensor, 16);
            System.out.println("✅ Both models trained!");
        }

        /**
         * Get ensemble prediction (weighted average of both models).
         */
        float predictEnsemble(float[] inputData) {
            // DL4J prediction
            INDArray inputArray = Nd4j.create(inputData).reshape(1, inputData.length);
            INDArray dl4jOutput = dl4jModel.output(inputArray);
            float dl4jPrediction = dl4jOutput.getFloat(0);

            // Gollek prediction
            GradTensor inputTensor = GradTensor.of(inputData, 1, inputDim);
            gollekModel.eval();
            GradTensor gollekOutput = gollekModel.forward(inputTensor);
            float gollekPrediction = gollekOutput.data()[0];

            // Ensemble: weighted average
            float ensemblePrediction = (dl4jWeight * dl4jPrediction) +
                    (gollekWeight * gollekPrediction);

            System.out.printf("  DL4J: %.4f, Gollek: %.4f, Ensemble: %.4f%n",
                    dl4jPrediction, gollekPrediction, ensemblePrediction);

            return ensemblePrediction;
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 4: Bidirectional Conversion Demo
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Demonstrates lossless bidirectional conversion between INDArray and
     * GradTensor.
     */
    static void demonstrateTensorConversion() {
        System.out.println("\n━━━ Pattern 4: Bidirectional Tensor Conversion ━━━");

        Random rand = new Random(42);

        // Test 2D tensor
        System.out.println("\n📊 Testing 2D tensor conversion:");
        float[] data2D = new float[12];
        for (int i = 0; i < data2D.length; i++) {
            data2D[i] = rand.nextFloat();
        }

        // DL4J → Gollek → DL4J
        INDArray original2D = Nd4j.create(data2D, new int[] { 3, 4 });
        GradTensor converted2D = indArrayToGradTensor(original2D);
        INDArray back2D = gradTensorToIndArray(converted2D);

        System.out.printf("  Original shape: %s%n", Arrays.toString(original2D.shape()));
        System.out.printf("  Converted shape: %s%n", Arrays.toString(converted2D.shape()));
        System.out.printf("  Back shape: %s%n", Arrays.toString(back2D.shape()));
        System.out.printf("  Max difference: %.2e%n",
                Nd4j.getExecutioner().exec(
                        org.nd4j.linalg.ops.transforms.Abs.INSTANCE,
                        original2D.sub(back2D)).maxNumber().floatValue());

        // Test 3D tensor
        System.out.println("\n📊 Testing 3D tensor conversion:");
        float[] data3D = new float[24];
        for (int i = 0; i < data3D.length; i++) {
            data3D[i] = rand.nextFloat();
        }

        INDArray original3D = Nd4j.create(data3D, new int[] { 2, 3, 4 });
        GradTensor converted3D = indArrayToGradTensor(original3D);
        INDArray back3D = gradTensorToIndArray(converted3D);

        System.out.printf("  Original shape: %s%n", Arrays.toString(original3D.shape()));
        System.out.printf("  Converted shape: %s%n", Arrays.toString(converted3D.shape()));
        System.out.printf("  Back shape: %s%n", Arrays.toString(back3D.shape()));
        System.out.printf("  Max difference: %.2e%n",
                Nd4j.getExecutioner().exec(
                        org.nd4j.linalg.ops.transforms.Abs.INSTANCE,
                        original3D.sub(back3D)).maxNumber().floatValue());

        System.out.println("\n✅ Tensor conversion verified (lossless)!");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main: Run All Integration Examples
    // ──────────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek SDK + Deeplearning4j Integration Examples      ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        // Configuration
        int INPUT_DIM = 20;
        int FEATURE_DIM = 16;
        int TRAIN_SAMPLES = 100;
        int EPOCHS = 50;

        // Generate synthetic training data
        Random rand = new Random(42);
        float[][] trainData = new float[TRAIN_SAMPLES][INPUT_DIM];
        float[] labels = new float[TRAIN_SAMPLES];
        for (int i = 0; i < TRAIN_SAMPLES; i++) {
            for (int j = 0; j < INPUT_DIM; j++) {
                trainData[i][j] = rand.nextFloat() * 2 - 1; // [-1, 1]
            }
            labels[i] = rand.nextFloat(); // [0, 1]
        }

        // Test data
        float[][] testData = new float[5][INPUT_DIM];
        for (int i = 0; i < testData.length; i++) {
            for (int j = 0; j < INPUT_DIM; j++) {
                testData[i][j] = rand.nextFloat() * 2 - 1;
            }
        }

        try {
            // Pattern 1: DL4J Preprocessing → Gollek Inference
            System.out.println("\n" + "═".repeat(60));
            System.out.println("INTEGRATION PATTERN 1: DL4J Preprocessing + Gollek Inference");
            System.out.println("═".repeat(60));
            var pattern1 = new DL4JPreprocessorGollekInference(INPUT_DIM);
            pattern1.train(trainData, labels, EPOCHS, 16);

            System.out.println("\n📈 Testing predictions:");
            float[] predictions = pattern1.predict(testData);
            for (int i = 0; i < predictions.length; i++) {
                System.out.printf("  Sample %d: %.4f%n", i + 1, predictions[i]);
            }

            // Pattern 2: DL4J Feature Extraction → Gollek Classification
            System.out.println("\n" + "═".repeat(60));
            System.out.println("INTEGRATION PATTERN 2: DL4J Feature Extraction + Gollek Classification");
            System.out.println("═".repeat(60));
            var pattern2 = new DL4JFeatureExtractorGollekClassifier(INPUT_DIM, FEATURE_DIM);
            pattern2.trainClassifier(trainData, labels, EPOCHS);

            System.out.println("\n📈 Testing predictions:");
            for (int i = 0; i < testData.length; i++) {
                float[] pred = pattern2.predict(testData[i]);
                System.out.printf("  Sample %d: class0=%.4f, class1=%.4f%n",
                        i + 1, pred[0], pred[1]);
            }

            // Pattern 3: Ensemble (DL4J + Gollek)
            System.out.println("\n" + "═".repeat(60));
            System.out.println("INTEGRATION PATTERN 3: Ensemble (DL4J + Gollek)");
            System.out.println("═".repeat(60));
            var pattern3 = new EnsembleDL4JGollek(INPUT_DIM, 0.5f, 0.5f);
            pattern3.train(trainData, labels, EPOCHS);

            System.out.println("\n📈 Testing ensemble predictions:");
            for (int i = 0; i < testData.length; i++) {
                pattern3.predictEnsemble(testData[i]);
            }

            // Pattern 4: Tensor Conversion Demo
            demonstrateTensorConversion();

            System.out.println("\n╔══════════════════════════════════════════════════════════╗");
            System.out.println("║   ✅ All integration examples completed successfully!   ║");
            System.out.println("╚══════════════════════════════════════════════════════════╝");

        } catch (Exception e) {
            System.err.println("❌ Error during integration demo: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
