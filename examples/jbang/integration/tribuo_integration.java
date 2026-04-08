///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS org.tribuo:tribuo-core:4.3.2
//DEPS org.tribuo:tribuo-classification-tree:4.3.2
//DEPS org.tribuo:tribuo-regression-tree:4.3.2
//DEPS org.tribuo:tribuo-data:4.3.2
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-nn-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-autograd-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-runtime-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-model-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-api-0.1.0-SNAPSHOT.jar

import org.tribuo.*;
import org.tribuo.impl.ArrayExample;
import org.tribuo.classification.*;
import org.tribuo.classification.dtree.CARTClassificationTrainer;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.common.tree.RandomForestTrainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.data.columnar.*;
import org.tribuo.data.columnar.processors.field.*;
import org.tribuo.data.columnar.processors.response.*;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.regression.*;
import org.tribuo.regression.rtree.CARTRegressionTrainer;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import java.time.OffsetDateTime;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;
import java.util.Locale;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.loss.MSELoss;
import tech.kayys.gollek.ml.nn.Trainer;
import tech.kayys.gollek.ml.nn.optim.Adam;

import java.io.*;
import java.net.URI;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

/**
 * Oracle Tribuo Integration with Gollek SDK.
 *
 * This example demonstrates how to integrate Gollek SDK with Oracle's Tribuo,
 * a machine learning library providing classification, regression, clustering,
 * anomaly detection, and regression tree algorithms.
 *
 * Use Cases:
 * - Using Tribuo's feature extraction pipelines
 * - Combining Tribuo classifiers with Gollek models
 * - Ensemble methods using both libraries
 * - Multi-model prediction aggregation
 * - Text classification with Tribuo NLP + Gollek NN
 * - Hybrid regression models
 *
 * Run: jbang tribuo_integration.java
 */
public class tribuo_integration {

    // ──────────────────────────────────────────────────────────────────────
    // Data Conversion Utilities
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Convert Tribuo MutableDataset to Gollek GradTensor.
     */
    static GradTensor datasetToTensor(MutableDataset<? extends Output<?>> dataset) {
        int numSamples = dataset.size();
        int featureDim = dataset.getFeatureIDMap().size();
        
        float[] data = new float[numSamples * featureDim];
        int idx = 0;
        
        for (Example<? extends Output<?>> example : dataset) {
            int j = 0;
            for (Feature f : example) {
                data[idx * featureDim + j++] = (float) f.getValue();
            }
            idx++;
        }
        
        return GradTensor.of(data, numSamples, featureDim);
    }

    /**
     * Convert float array to Tribuo Example.
     */
    static Example<Label> toTribuoExample(float[] features, String labelName) {
        String[] featureNames = new String[features.length];
        double[] featureValues = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            featureNames[i] = "f" + i;
            featureValues[i] = features[i];
        }
        return new ArrayExample<>(new Label(labelName), featureNames, featureValues);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 1: Tribuo Feature Extraction → Gollek Classification
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Uses Tribuo's feature processing pipeline to extract features,
     * then classifies with Gollek neural network.
     */
    static class TribuoFeaturePipelineGollekClassifier {
        private final FieldResponseProcessor<Label> responseProcessor;
        private final FieldProcessor[] fieldProcessors;
        private final RowProcessor<Label> rowProcessor;
        private final tech.kayys.gollek.ml.nn.Module gollekClassifier;
        private final int featureDim;
        private FeatureMap featureMap;

        TribuoFeaturePipelineGollekClassifier(int featureDim) {
            this.featureDim = featureDim;
            this.responseProcessor = new FieldResponseProcessor<>("CLASS", "UNKNOWN", new LabelFactory());
            Map<String, FieldProcessor> fpMap = new HashMap<>();
            for (int i = 0; i < featureDim; i++) {
                fpMap.put("f" + i, new DoubleFieldProcessor("f" + i));
            }
            this.fieldProcessors = fpMap.values().toArray(new FieldProcessor[0]);
            this.rowProcessor = new RowProcessor<>(responseProcessor, fpMap);
            
            this.gollekClassifier = new tech.kayys.gollek.ml.nn.Sequential(
                new Linear(featureDim, 64),
                new ReLU(),
                new Dropout(0.2f),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            System.out.println("📦 Tribuo Feature Pipeline + Gollek Classifier");
            System.out.printf("   Features: %d → 64 → 32 → 2%n", featureDim);
        }

        /**
         * Train on Tribuo dataset, then train Gollek.
         */
        void train(MutableDataset<Label> tribuoDataset, int epochs, int batchSize) {
            System.out.println("\n━━━ Pattern: Tribuo Feature Pipeline → Gollek Classification ━━━");

            // Extract feature map from Tribuo dataset
            this.featureMap = tribuoDataset.getFeatureIDMap();
            System.out.printf("📊 Tribuo feature map: %d features%n", featureMap.size());

            // Convert Tribuo dataset to Gollek tensor
            GradTensor inputTensor = datasetToTensor(tribuoDataset);
            
            // Create label tensor
            float[] labelData = new float[(int) tribuoDataset.size()];
            int sampleCount = 0;
            for (Example<Label> example : tribuoDataset) {
                labelData[sampleCount++] = example.getOutput().getLabel().equals("CLASS_1") ? 1f : 0f;
            }
            GradTensor labelTensor = GradTensor.of(labelData, (int) tribuoDataset.size());

            // Train Gollek
            System.out.println("🏋️ Training Gollek classifier...");
            var trainer = Trainer.builder()
                    .model(gollekClassifier)
                    .optimizer(new Adam(gollekClassifier.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
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

            trainer.fit(inputTensor, labelTensor, batchSize);
            System.out.println("✅ Training complete!");
        }

        /**
         * Predict using Gollek.
         */
        GollekPrediction predict(float[] features) {
            gollekClassifier.eval();
            GradTensor input = GradTensor.of(features, 1, featureDim);
            GradTensor output = gollekClassifier.forward(input);
            float[] scores = output.data();

            // Softmax
            float maxScore = Math.max(scores[0], scores[1]);
            float exp0 = (float) Math.exp(scores[0] - maxScore);
            float exp1 = (float) Math.exp(scores[1] - maxScore);
            float sum = exp0 + exp1;

            int predictedClass = scores[0] > scores[1] ? 0 : 1;
            return new GollekPrediction(predictedClass, new float[]{exp0 / sum, exp1 / sum});
        }
    }

    static class GollekPrediction {
        final int predictedClass;
        final float[] probabilities;

        GollekPrediction(int predictedClass, float[] probabilities) {
            this.predictedClass = predictedClass;
            this.probabilities = probabilities;
        }

        @Override
        public String toString() {
            return String.format("Class=%d, Prob=[%.3f, %.3f], Confidence=%.1f%%",
                    predictedClass, probabilities[0], probabilities[1],
                    Math.max(probabilities[0], probabilities[1]) * 100);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 2: Tribuo XGBoost + Gollek Ensemble
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Ensemble combining Tribuo's XGBoost classifier with Gollek neural network.
     * XGBoost excels at tabular data, while NN captures complex patterns.
     */
    static class EnsembleTribuoRFGollek {
        private Model<Label> tribuoModel;
        private final tech.kayys.gollek.ml.nn.Module gollekModel;
        private final int featureDim;
        private final double tribuoWeight;
        private final double gollekWeight;

        EnsembleTribuoRFGollek(int featureDim, double tribuoWeight, double gollekWeight) {
            this.featureDim = featureDim;
            this.tribuoWeight = tribuoWeight;
            this.gollekWeight = gollekWeight;
            
            this.gollekModel = new tech.kayys.gollek.ml.nn.Sequential(
                new Linear(featureDim, 64),
                new ReLU(),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            System.out.println("🎭 Ensemble: Tribuo Random Forest + Gollek Neural Network");
            System.out.printf("   Weights: Tribuo=%.1f%%, Gollek=%.1f%%%n", 
                    tribuoWeight * 100, gollekWeight * 100);
        }

        /**
         * Train both models.
         */
        void train(MutableDataset<Label> dataset, int epochs) {
            System.out.println("\n━━━ Pattern: Ensemble (Tribuo RF + Gollek NN) ━━━");

            // Train Tribuo RF
            System.out.println("🌲 Training Tribuo Random Forest...");
            var leafTrainer = new CARTClassificationTrainer(5, 0.7f, false, 1L);
            var rfTrainer = new RandomForestTrainer<>(leafTrainer, new VotingCombiner(), 20);
            tribuoModel = rfTrainer.train(dataset);
            System.out.println("✅ Random Forest trained!");

            // Train Gollek
            System.out.println("🏋️ Training Gollek Neural Network...");
            GradTensor inputTensor = datasetToTensor(dataset);
            
            float[] labelData = new float[dataset.size()];
            int count = 0;
            for (Example<Label> example : dataset) {
                labelData[count++] = example.getOutput().getLabel().equals("CLASS_1") ? 1f : 0f;
            }
            GradTensor labelTensor = GradTensor.of(labelData, (int) dataset.size());

            var gollekTrainer = Trainer.builder()
                    .model(gollekModel)
                    .optimizer(new Adam(gollekModel.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
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

            gollekTrainer.fit(inputTensor, labelTensor, 16);
            System.out.println("✅ Neural Network trained!");
        }

        /**
         * Predict using ensemble voting.
         */
        EnsembleResult predict(float[] features) {
            // Tribuo prediction
            Example<Label> example = toTribuoExample(features, "CLASS_0");
            org.tribuo.Prediction<Label> prediction = tribuoModel.predict(example);
            
            // Normalize Tribuo probabilities
            Map<String, Float> tribuoProbs = new HashMap<>();
            for (Map.Entry<String, Label> entry : prediction.getOutputScores().entrySet()) {
                tribuoProbs.put(entry.getKey(), (float) entry.getValue().getScore());
            }
            
            float tribuoProb0 = tribuoProbs.getOrDefault("CLASS_0", 0f);
            float tribuoProb1 = tribuoProbs.getOrDefault("CLASS_1", 0f);
            float tribuoTotal = tribuoProb0 + tribuoProb1;
            if (tribuoTotal > 0) {
                tribuoProb0 /= tribuoTotal;
                tribuoProb1 /= tribuoTotal;
            }
            int tribuoClass = tribuoProb0 > tribuoProb1 ? 0 : 1;

            // Gollek prediction
            gollekModel.eval();
            GradTensor input = GradTensor.of(features, 1, featureDim);
            GradTensor output = gollekModel.forward(input);
            float[] gollekScores = output.data();
            
            // Softmax
            float maxScore = Math.max(gollekScores[0], gollekScores[1]);
            float exp0 = (float) Math.exp(gollekScores[0] - maxScore);
            float exp1 = (float) Math.exp(gollekScores[1] - maxScore);
            float sum = exp0 + exp1;
            float[] gollekProbs = new float[]{exp0 / sum, exp1 / sum};
            int gollekClass = gollekProbs[0] > gollekProbs[1] ? 0 : 1;

            // Ensemble: weighted average
            float ensembleProb0 = (float) (tribuoWeight * tribuoProb0 + gollekWeight * gollekProbs[0]);
            float ensembleProb1 = (float) (tribuoWeight * tribuoProb1 + gollekWeight * gollekProbs[1]);
            int ensembleClass = ensembleProb0 > ensembleProb1 ? 0 : 1;

            return new EnsembleResult(ensembleClass, ensembleProb0, ensembleProb1,
                    tribuoClass, gollekClass, tribuoProbs, gollekProbs);
        }


    }

    static class EnsembleResult {
        final int ensembleClass;
        final float ensembleProb0;
        final float ensembleProb1;
        final int tribuoClass;
        final int gollekClass;
        final Map<String, Float> tribuoProbs;
        final float[] gollekProbs;

        EnsembleResult(int ensembleClass, float ensembleProb0, float ensembleProb1,
                      int tribuoClass, int gollekClass, Map<String, Float> tribuoProbs, float[] gollekProbs) {
            this.ensembleClass = ensembleClass;
            this.ensembleProb0 = ensembleProb0;
            this.ensembleProb1 = ensembleProb1;
            this.tribuoClass = tribuoClass;
            this.gollekClass = gollekClass;
            this.tribuoProbs = tribuoProbs;
            this.gollekProbs = gollekProbs;
        }

        @Override
        public String toString() {
            return String.format("Ensemble=%d (%.1f%%), Tribuo=%d, Gollek=%d",
                    ensembleClass, Math.max(ensembleProb0, ensembleProb1) * 100,
                    tribuoClass, gollekClass);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 3: Tribuo Text Classification + Gollek
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Uses Tribuo's text processing pipeline for NLP feature extraction,
     * then classifies with Gollek neural network.
     */
    static class TribuoTextPipelineGollekClassifier {
        private final BasicPipeline textPipeline;
        private tech.kayys.gollek.ml.nn.Module gollekClassifier;
        private int vocabSize;
        private Map<String, Integer> vocabIndex;

        TribuoTextPipelineGollekClassifier(int vocabSize) {
            this.vocabSize = vocabSize;
            this.textPipeline = new BasicPipeline(new BreakIteratorTokenizer(Locale.US), 1);
            
            this.gollekClassifier = new tech.kayys.gollek.ml.nn.Sequential(
                new Linear(vocabSize, 64),
                new ReLU(),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            System.out.println("📝 Tribuo Text Pipeline + Gollek Classifier");
            System.out.printf("   Vocabulary size: %d, Architecture: %d → 64 → 32 → 2%n", vocabSize, vocabSize);
        }

        /**
         * Train text pipeline and Gollek classifier.
         */
        void train(List<String> texts, List<String> labels, int epochs) {
            System.out.println("\n━━━ Pattern: Tribuo Text Pipeline → Gollek Classification ━━━");

            // Create Tribuo text dataset
            MutableDataset<Label> dataset = new MutableDataset<>(
                new SimpleDataSourceProvenance("synthetic", OffsetDateTime.now(), new LabelFactory()),
                new LabelFactory()
            );
            for (int i = 0; i < texts.size(); i++) {
                List<Feature> extractedFeatures = textPipeline.process("text", texts.get(i));
                Example<Label> example = new ArrayExample<>(new Label(labels.get(i)), extractedFeatures);
                dataset.add(example);
            }

            // Get vocabulary/feature map
            this.vocabIndex = new HashMap<>();
            int idx = 0;
            for (String feature : dataset.getFeatureIDMap().keySet()) {
                vocabIndex.put(feature, idx++);
            }
            int actualVocab = Math.min(vocabSize, vocabIndex.size());
            System.out.printf("📊 Vocabulary: %d features extracted%n", actualVocab);

            // Convert to Gollek tensor
            float[] featureData = new float[dataset.size() * actualVocab];
            int sampleIdx = 0;
            for (Example<Label> example : dataset) {
                for (Feature f : example) {
                    Integer featureIdx = vocabIndex.get(f.getName());
                    if (featureIdx != null && featureIdx < actualVocab) {
                        featureData[sampleIdx * actualVocab + featureIdx] = (float) f.getValue();
                    }
                }
                sampleIdx++;
            }
            GradTensor inputTensor = GradTensor.of(featureData, dataset.size(), actualVocab);

            // Label tensor
            float[] labelData = new float[dataset.size()];
            sampleIdx = 0;
            for (Example<Label> example : dataset) {
                labelData[sampleIdx++] = example.getOutput().getLabel().equals("POSITIVE") ? 1f : 0f;
            }
            GradTensor labelTensor = GradTensor.of(labelData, (int) dataset.size());

            // 3. Dynamic initialization based on actual vocabulary size
            this.vocabSize = (int) inputTensor.shape()[1];
            System.out.printf("📊 Vocabulary: %d features extracted%n", this.vocabSize);
            
            this.gollekClassifier = new tech.kayys.gollek.ml.nn.Sequential(
                new Linear(this.vocabSize, 64),
                new ReLU(),
                new Dropout(0.2f),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 2)
            );

            // 4. Train Gollek
            System.out.println("🏋️ Training Gollek classifier...");
            var trainer = Trainer.builder()
                    .model(gollekClassifier)
                    .optimizer(new Adam(gollekClassifier.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
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

            trainer.fit(inputTensor, labelTensor, 8);
            System.out.println("✅ Training complete!");
        }

        /**
         * Predict sentiment for text.
         */
        GollekPrediction predict(String text) {
            // Process text with Tribuo pipeline
            List<Feature> extractedFeatures = textPipeline.process("text", text);
            Example<Label> example = new ArrayExample<>(new Label("UNKNOWN"), extractedFeatures);
            
            // Convert to feature vector
            float[] features = new float[vocabSize];
            for (Feature f : example) {
                Integer idx = vocabIndex.get(f.getName());
                if (idx != null && idx < vocabSize) {
                    features[idx] = (float) f.getValue();
                }
            }

            // Gollek inference
            gollekClassifier.eval();
            GradTensor input = GradTensor.of(features, 1, vocabSize);
            GradTensor output = gollekClassifier.forward(input);
            float[] scores = output.data();

            // Softmax
            float maxScore = Math.max(scores[0], scores[1]);
            float exp0 = (float) Math.exp(scores[0] - maxScore);
            float exp1 = (float) Math.exp(scores[1] - maxScore);
            float sum = exp0 + exp1;

            int predictedClass = scores[0] > scores[1] ? 0 : 1;
            return new GollekPrediction(predictedClass, new float[]{exp0 / sum, exp1 / sum});
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern 4: Tribuo Regression + Gollek Refinement
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Uses Tribuo's regression model for initial prediction,
     * then Gollek for refinement/residual learning.
     */
    static class TribuoRegressionGollekRefinement {
        private CARTRegressionTrainer tribuoTrainer;
        private Model<Regressor> tribuoModel;
        private final tech.kayys.gollek.ml.nn.Module gollekRefiner;
        private final int featureDim;

        TribuoRegressionGollekRefinement(int featureDim) {
            this.featureDim = featureDim;
            this.tribuoTrainer = new CARTRegressionTrainer(10); // maxDepth
            
            // Gollek refiner: learns residuals
            this.gollekRefiner = new tech.kayys.gollek.ml.nn.Sequential(
                new Linear(featureDim + 1, 32), // features + tribuo prediction
                new ReLU(),
                new Linear(32, 1)
            );

            System.out.println("📈 Tribuo Regression + Gollek Refinement");
            System.out.printf("   Features: %d, Tribuo predicts → Gollek refines residuals%n", featureDim);
        }

        /**
         * Train Tribuo, then train Gollek on residuals.
         */
        void train(MutableDataset<Regressor> dataset, int epochs) {
            System.out.println("\n━━━ Pattern: Tribuo Regression + Gollek Residual Refinement ━━━");

            // Train Tribuo regression
            System.out.println("🌲 Training Tribuo CART Regression...");
            tribuoModel = tribuoTrainer.train(dataset);
            System.out.println("✅ Tribuo regression trained!");

            // Calculate residuals and train Gollek
            System.out.println("🏋️ Training Gollek on residuals...");
            
            float[] residualData = new float[dataset.size() * (featureDim + 1)];
            float[] residualTargets = new float[dataset.size()];
            
            int sampleIdx = 0;
            for (Example<Regressor> example : dataset) {
                float[] features = new float[featureDim];
                int featIdx = 0;
                for (Feature f : example) {
                    features[featIdx++] = (float) f.getValue();
                }
                
                // Tribuo prediction
                org.tribuo.Prediction<Regressor> prediction = tribuoModel.predict(example);
                float tribuoValue = (float) prediction.getOutput().getValues()[0];
                
                // Actual value
                float actualValue = (float) example.getOutput().getValues()[0];
                
                // Residual
                float residual = actualValue - tribuoValue;
                
                // Input to Gollek: features + tribuo prediction
                for (int i = 0; i < featureDim; i++) {
                    residualData[sampleIdx * (featureDim + 1) + i] = features[i];
                }
                residualData[sampleIdx * (featureDim + 1) + featureDim] = tribuoValue;
                residualTargets[sampleIdx] = residual;
                
                sampleIdx++;
            }

            GradTensor inputTensor = GradTensor.of(residualData, (int) dataset.size(), featureDim + 1);
            // Explicitly create [N, 1] tensor for targets
            GradTensor targetTensor = GradTensor.of(residualTargets, (int) dataset.size(), 1);
            targetTensor = targetTensor.reshape((int) dataset.size(), 1);

            var gollekTrainer = Trainer.builder()
                    .model(gollekRefiner)
                    .optimizer(new Adam(gollekRefiner.parameters(), 0.01f))
                    .lossFunction((pred, target) -> new MSELoss().compute(pred, target.reshape(pred.shape())))
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

            gollekTrainer.fit(inputTensor, targetTensor, 16);
            System.out.println("✅ Gollek refiner trained!");
        }

        /**
         * Predict using Tribuo + Gollek refinement.
         */
        RegressionResult predict(float[] features) {
            // Create Tribuo example
            String[] featureNames = new String[features.length];
            double[] featureValues = new double[features.length];
            for (int i = 0; i < features.length; i++) {
                featureNames[i] = "f" + i;
                featureValues[i] = features[i];
            }
            Example<Regressor> example = new ArrayExample<>(new Regressor("target", 0.0), featureNames, featureValues);

            // Tribuo prediction
            org.tribuo.Prediction<Regressor> prediction = tribuoModel.predict(example);
            float tribuoValue = (float) prediction.getOutput().getValues()[0];

            // Gollek refinement
            gollekRefiner.eval();
            float[] gollekInput = new float[features.length + 1];
            System.arraycopy(features, 0, gollekInput, 0, features.length);
            gollekInput[features.length] = tribuoValue;
            
            GradTensor input = GradTensor.of(gollekInput, 1, features.length + 1);
            GradTensor residual = gollekRefiner.forward(input);
            float refinedValue = tribuoValue + residual.data()[0];

            return new RegressionResult(tribuoValue, refinedValue);
        }


    }

    static class RegressionResult {
        final float tribuoPrediction;
        final float refinedPrediction;

        RegressionResult(float tribuoPrediction, float refinedPrediction) {
            this.tribuoPrediction = tribuoPrediction;
            this.refinedPrediction = refinedPrediction;
        }

        @Override
        public String toString() {
            return String.format("Tribuo=%.4f, Refined=%.4f", tribuoPrediction, refinedPrediction);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Demo Dataset Generator
    // ──────────────────────────────────────────────────────────────────────

    static class DemoDataset {
        static MutableDataset<Label> generateClassificationData(int numSamples, int numFeatures, Random rand) {
            MutableDataset<Label> dataset = new MutableDataset<>(
                new SimpleDataSourceProvenance("synthetic", OffsetDateTime.now(), new LabelFactory()),
                new LabelFactory()
            );
            String[] featureNames = new String[numFeatures];
            for (int i = 0; i < numFeatures; i++) featureNames[i] = "f" + i;

            for (int i = 0; i < numSamples; i++) {
                double[] featureValues = new double[numFeatures];
                float sum = 0;
                for (int j = 0; j < numFeatures; j++) {
                    double value = rand.nextGaussian();
                    featureValues[j] = value;
                    sum += value;
                }
                
                String labelName = sum > 0 ? "CLASS_1" : "CLASS_0";
                dataset.add(new ArrayExample<>(new Label(labelName), featureNames, featureValues));
            }

            return dataset;
        }

        static MutableDataset<Regressor> generateRegressionData(int numSamples, int numFeatures, Random rand) {
            MutableDataset<Regressor> dataset = new MutableDataset<>(
                new SimpleDataSourceProvenance("synthetic", OffsetDateTime.now(), new org.tribuo.regression.RegressionFactory()),
                new org.tribuo.regression.RegressionFactory()
            );
            String[] featureNames = new String[numFeatures];
            for (int i = 0; i < numFeatures; i++) featureNames[i] = "f" + i;
            
            for (int i = 0; i < numSamples; i++) {
                double[] featureValues = new double[numFeatures];
                double target = 0;
                for (int j = 0; j < numFeatures; j++) {
                    double value = rand.nextGaussian();
                    featureValues[j] = value;
                    target += value * (j + 1) * 0.1;
                }
                
                dataset.add(new ArrayExample<>(new Regressor("target", target), featureNames, featureValues));
            }

            return dataset;
        }

        static List<String> getSampleTexts() {
            return Arrays.asList(
                "This movie is absolutely fantastic and I loved every moment",
                "The acting was superb and the storyline kept me engaged",
                "A masterpiece of modern cinema with brilliant performances",
                "This was a complete waste of time and money",
                "The plot made no sense and the acting was terrible",
                "I could not wait for this boring movie to end",
                "Highly recommended for anyone who enjoys quality entertainment",
                "Avoid this film at all costs it is dreadful",
                "The director did an amazing job bringing this story to life",
                "The worst movie I have seen in years"
            );
        }

        static List<String> getSampleLabels() {
            return Arrays.asList(
                "POSITIVE", "POSITIVE", "POSITIVE",
                "NEGATIVE", "NEGATIVE", "NEGATIVE",
                "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main: Run All Integration Examples
    // ──────────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek SDK + Oracle Tribuo Integration Examples       ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        Random rand = new Random(42);
        int NUM_SAMPLES = 200;
        int NUM_FEATURES = 10;
        int EPOCHS = 50;

        try {
            // Pattern 1: Tribuo Feature Pipeline → Gollek Classification
            System.out.println("\n" + "═".repeat(60));
            System.out.println("PATTERN 1: Tribuo Feature Pipeline → Gollek Classification");
            System.out.println("═".repeat(60));
            
            MutableDataset<Label> trainDataset = DemoDataset.generateClassificationData(NUM_SAMPLES, NUM_FEATURES, rand);
            
            var pattern1 = new TribuoFeaturePipelineGollekClassifier(NUM_FEATURES);
            pattern1.train(trainDataset, EPOCHS, 16);

            // Test
            MutableDataset<Label> testDataset = DemoDataset.generateClassificationData(20, NUM_FEATURES, rand);
            int correct = 0;
            for (Example<Label> example : testDataset) {
                float[] features = new float[NUM_FEATURES];
                int j = 0;
                for (Feature f : example) {
                    int idx = Integer.parseInt(f.getName().substring(1));
                    features[idx] = (float) f.getValue();
                }
                GollekPrediction pred = pattern1.predict(features);
                int actual = example.getOutput().getLabel().equals("CLASS_1") ? 1 : 0;
                if (pred.predictedClass == actual) correct++;
            }
            System.out.printf("\n📈 Test Accuracy: %d/%d (%.1f%%)%n", 
                    correct, testDataset.size(), 100.0 * correct / testDataset.size());

            // Pattern 2: Ensemble (Tribuo Random Forest + Gollek)
            System.out.println("\n" + "═".repeat(60));
            System.out.println("PATTERN 2: Ensemble (Tribuo Random Forest + Gollek NN)");
            System.out.println("═".repeat(60));
            
            var pattern2 = new EnsembleTribuoRFGollek(NUM_FEATURES, 0.5, 0.5);
            pattern2.train(trainDataset, EPOCHS);

            // Test with detailed results
            MutableDataset<Label> testDataset2 = DemoDataset.generateClassificationData(5, NUM_FEATURES, rand);
            System.out.println("\n📈 Testing ensemble predictions:");
            for (Example<Label> example : testDataset2) {
                float[] features = new float[NUM_FEATURES];
                for (Feature f : example) {
                    int idx = Integer.parseInt(f.getName().substring(1));
                    features[idx] = (float) f.getValue();
                }
                EnsembleResult result = pattern2.predict(features);
                int actual = example.getOutput().getLabel().equals("CLASS_1") ? 1 : 0;
                System.out.printf("  %s (Actual: %d) %s%n", 
                        result, actual, result.ensembleClass == actual ? "✓" : "✗");
            }

            // Pattern 3: Tribuo Text Pipeline → Gollek
            System.out.println("\n" + "═".repeat(60));
            System.out.println("PATTERN 3: Tribuo Text Pipeline → Gollek Sentiment Analysis");
            System.out.println("═".repeat(60));
            
            int vocabSize = 100;
            var pattern3 = new TribuoTextPipelineGollekClassifier(vocabSize);
            pattern3.train(DemoDataset.getSampleTexts(), DemoDataset.getSampleLabels(), 50);

            // Test
            List<String> testTexts = Arrays.asList(
                "Excellent film with outstanding performances",
                "Terrible movie I hated everything about it",
                "Not bad but could have been better"
            );
            System.out.println("\n📈 Testing sentiment analysis:");
            for (String text : testTexts) {
                GollekPrediction pred = pattern3.predict(text);
                System.out.printf("  Text: \"%s\"%n", text);
                System.out.printf("  → %s%n", pred);
            }

            MutableDataset<Regressor> regressionData = DemoDataset.generateRegressionData(NUM_SAMPLES, NUM_FEATURES, rand);
            
            var pattern4 = new TribuoRegressionGollekRefinement(NUM_FEATURES);
            pattern4.train(regressionData, EPOCHS);

            // Test
            MutableDataset<Regressor> testData4 = DemoDataset.generateRegressionData(10, NUM_FEATURES, rand);
            System.out.println("\n📈 Testing regression refinement:");
            for (Example<Regressor> example : testData4) {
                float[] features = new float[NUM_FEATURES];
                for (Feature f : example) {
                    int idx = Integer.parseInt(f.getName().substring(1));
                    features[idx] = (float) f.getValue();
                }
                RegressionResult result = pattern4.predict(features);
                float actual = (float) example.getOutput().getValues()[0];
                System.out.printf("  Actual: %.4f, %s (Error reduction: %.1f%%)%n", 
                        actual, result, 100.0 * (Math.abs(actual - result.tribuoPrediction) - Math.abs(actual - result.refinedPrediction)) / Math.max(1e-6, Math.abs(actual - result.tribuoPrediction)));
            }
            MutableDataset<Regressor> testData = DemoDataset.generateRegressionData(5, NUM_FEATURES, rand);
            System.out.println("\n📈 Testing regression predictions:");
            for (Example<Regressor> example : testData) {
                float[] features = new float[NUM_FEATURES];
                for (Feature f : example) {
                    int idx = Integer.parseInt(f.getName().substring(1));
                    features[idx] = (float) f.getValue();
                }
                RegressionResult result = pattern4.predict(features);
                float actual = (float) example.getOutput().getValues()[0];
                System.out.printf("  Actual: %.4f, %s%n", actual, result);
            }

            System.out.println("\n╔══════════════════════════════════════════════════════════╗");
            System.out.println("║   ✅ All Tribuo integration examples completed!         ║");
            System.out.println("╚══════════════════════════════════════════════════════════╝");

        } catch (Exception e) {
            System.err.println("❌ Error during integration demo: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
