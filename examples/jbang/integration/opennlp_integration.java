///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS org.apache.opennlp:opennlp-tools:2.3.3
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-nn-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-autograd-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-runtime-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-model-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-api-0.1.0-SNAPSHOT.jar
//DEPS com.google.code.gson:gson:2.11.0
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import opennlp.tools.namefind.*;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.*;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.optim.Adam;

import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

/**
 * Apache OpenNLP Integration with Gollek SDK.
 *
 * This example demonstrates how to integrate Gollek SDK with Apache OpenNLP,
 * a machine learning based toolkit for processing natural language text.
 *
 * Use Cases:
 * - Text preprocessing and tokenization
 * - Named entity recognition (NER)
 * - Sentence detection and segmentation
 * - Part-of-speech tagging
 * - Combining OpenNLP tools with Gollek inference
 * - Building custom text classification pipelines
 *
 * Run: jbang opennlp_integration.java
 *
 * Note: This example downloads OpenNLP models on first run. Models are cached
 * in the system temp directory.
 */
public class opennlp_integration {

    // ──────────────────────────────────────────────────────────────────────
    // Model Manager - Downloads and caches OpenNLP models
    // ──────────────────────────────────────────────────────────────────────

    static class ModelManager {
        private static final String MODEL_CACHE_DIR = System.getProperty("java.io.tmpdir") + "/opennlp-models";
        
        private static final Map<String, String> MODEL_URLS = Map.of(
            "sentence-en", "https://opennlp.sourceforge.net/models-1.5/en-sent.bin",
            "tokenizer-en", "https://opennlp.sourceforge.net/models-1.5/en-token.bin",
            "pos-en", "https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin",
            "ner-person-en", "https://opennlp.sourceforge.net/models-1.5/en-ner-person.bin",
            "ner-location-en", "https://opennlp.sourceforge.net/models-1.5/en-ner-location.bin",
            "ner-organization-en", "https://opennlp.sourceforge.net/models-1.5/en-ner-organization.bin"
        );

        static Path getModelPath(String modelName) throws IOException {
            Path cacheDir = Paths.get(MODEL_CACHE_DIR);
            if (!Files.exists(cacheDir)) {
                Files.createDirectories(cacheDir);
            }

            String url = MODEL_URLS.get(modelName);
            if (url == null) {
                throw new IllegalArgumentException("Unknown model: " + modelName);
            }

            Path modelPath = cacheDir.resolve(modelName + ".bin");
            
            if (!Files.exists(modelPath)) {
                System.out.printf("📥 Downloading OpenNLP model: %s...%n", modelName);
                try (InputStream in = java.net.URI.create(url).toURL().openStream()) {
                    Files.copy(in, modelPath, StandardCopyOption.REPLACE_EXISTING);
                }
                System.out.printf("✅ Model cached: %s%n", modelPath);
            } else {
                System.out.printf("📦 Using cached model: %s%n", modelName);
            }

            return modelPath;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // OpenNLP Pipeline - Comprehensive Text Processing
    // ──────────────────────────────────────────────────────────────────────

    /**
     * A comprehensive OpenNLP processing pipeline that provides:
     * - Sentence detection
     * - Tokenization
     * - POS tagging
     * - Named entity recognition
     */
    static class OpenNLPPipeline implements AutoCloseable {
        private final SentenceDetectorME sentenceDetector;
        private final TokenizerME tokenizer;
        private final POSTaggerME posTagger;
        private final NameFinderME[] nameFinders;
        private final Map<String, Integer> featureIndex;
        private final int featureDim;

        OpenNLPPipeline() throws IOException {
            // Load models
            SentenceModel sentModel = new SentenceModel(
                Files.newInputStream(ModelManager.getModelPath("sentence-en")));
            TokenizerModel tokenModel = new TokenizerModel(
                Files.newInputStream(ModelManager.getModelPath("tokenizer-en")));
            POSModel posModel = new POSModel(
                Files.newInputStream(ModelManager.getModelPath("pos-en")));

            // Initialize tools
            this.sentenceDetector = new SentenceDetectorME(sentModel);
            this.tokenizer = new TokenizerME(tokenModel);
            this.posTagger = new POSTaggerME(posModel);

            // Load NER models
            this.nameFinders = new NameFinderME[] {
                new NameFinderME(new TokenNameFinderModel(
                    Files.newInputStream(ModelManager.getModelPath("ner-person-en")))),
                new NameFinderME(new TokenNameFinderModel(
                    Files.newInputStream(ModelManager.getModelPath("ner-location-en")))),
                new NameFinderME(new TokenNameFinderModel(
                    Files.newInputStream(ModelManager.getModelPath("ner-organization-en"))))
            };

            // Define feature dimensions for Gollek model
            // 1. Sentence count
            // 2. Token count
            // 3. Average token length
            // 4. Noun ratio
            // 5. Verb ratio
            // 6. Adjective ratio
            // 7. Adverb ratio
            // 8. Named entity count (total)
            // 9. Person entity count
            // 10. Location entity count
            // 11. Organization entity count
            // 12. Unique token ratio (vocabulary richness)
            // 13. Average sentence length
            // 14. Question mark presence
            // 15. Exclamation mark presence
            this.featureDim = 15;

            this.featureIndex = new HashMap<>();
            for (int i = 0; i < featureDim; i++) {
                featureIndex.put("feature_" + i, i);
            }

            System.out.println("📦 OpenNLP Pipeline initialized");
            System.out.printf("   Feature dimension: %d%n", featureDim);
        }

        /**
         * Process text and extract features for Gollek model.
         */
        float[] extractFeatures(String text) {
            float[] features = new float[featureDim];
            int idx = 0;

            // Sentence detection
            String[] sentences = sentenceDetector.sentDetect(text);
            features[idx++] = Math.min(1.0f, sentences.length / 10.0f); // Normalized sentence count

            // Tokenization
            List<String> allTokens = new ArrayList<>();
            for (String sentence : sentences) {
                allTokens.addAll(Arrays.asList(tokenizer.tokenize(sentence)));
            }
            String[] tokens = allTokens.toArray(new String[0]);

            // Token count (normalized)
            features[idx++] = Math.min(1.0f, tokens.length / 50.0f);

            // Average token length
            if (tokens.length > 0) {
                double avgLen = Arrays.stream(tokens)
                        .mapToInt(String::length)
                        .average()
                        .orElse(0.0);
                features[idx++] = (float) Math.min(1.0, avgLen / 10.0);
            } else {
                features[idx++] = 0;
            }

            // POS tagging
            String[] posTags = posTagger.tag(tokens);
            Map<String, Integer> posCounts = new HashMap<>();
            for (String tag : posTags) {
                posCounts.merge(tag, 1, Integer::sum);
            }

            // POS ratios
            // Nouns: NN, NNS, NNP, NNPS
            float nounCount = getCountForTags(posCounts, "NN") / (float) Math.max(1, tokens.length);
            features[idx++] = nounCount;

            // Verbs: VB, VBD, VBG, VBN, VBP, VBZ
            float verbCount = getCountForTags(posCounts, "VB") / (float) Math.max(1, tokens.length);
            features[idx++] = verbCount;

            // Adjectives: JJ, JJR, JJS
            float adjCount = getCountForTags(posCounts, "JJ") / (float) Math.max(1, tokens.length);
            features[idx++] = adjCount;

            // Adverbs: RB, RBR, RBS
            float advCount = getCountForTags(posCounts, "RB") / (float) Math.max(1, tokens.length);
            features[idx++] = advCount;

            // Named Entity Recognition
            int totalNE = 0, personNE = 0, locationNE = 0, orgNE = 0;

            for (NameFinderME finder : nameFinders) {
                Span[] spans = finder.find(tokens);
                totalNE += spans.length;
                
                for (Span span : spans) {
                    String type = span.getType();
                    if ("person".equals(type)) personNE++;
                    else if ("location".equals(type)) locationNE++;
                    else if ("organization".equals(type)) orgNE++;
                }

                finder.clearAdaptiveData();
            }

            features[idx++] = Math.min(1.0f, totalNE / 10.0f);
            features[idx++] = Math.min(1.0f, personNE / 5.0f);
            features[idx++] = Math.min(1.0f, locationNE / 5.0f);
            features[idx++] = Math.min(1.0f, orgNE / 5.0f);

            // Vocabulary richness (unique token ratio)
            long uniqueTokens = Arrays.stream(tokens).distinct().count();
            features[idx++] = (float) uniqueTokens / Math.max(1, tokens.length);

            // Average sentence length
            if (sentences.length > 0) {
                features[idx++] = Math.min(1.0f, (tokens.length / (float) sentences.length) / 30.0f);
            } else {
                features[idx++] = 0;
            }

            // Punctuation features
            features[idx++] = text.contains("?") ? 1.0f : 0.0f;
            features[idx++] = text.contains("!") ? 1.0f : 0.0f;

            return features;
        }

        private float getCountForTags(Map<String, Integer> posCounts, String prefix) {
            return posCounts.entrySet().stream()
                    .filter(e -> e.getKey().startsWith(prefix))
                    .mapToInt(Map.Entry::getValue)
                    .sum();
        }

        /**
         * Get detailed NLP analysis result.
         */
        AnalysisResult analyze(String text) {
            String[] sentences = sentenceDetector.sentDetect(text);
            List<String> allTokens = new ArrayList<>();
            for (String sentence : sentences) {
                allTokens.addAll(Arrays.asList(tokenizer.tokenize(sentence)));
            }
            String[] tokens = allTokens.toArray(new String[0]);
            String[] posTags = posTagger.tag(tokens);

            List<NamedEntity> entities = new ArrayList<>();
            for (NameFinderME finder : nameFinders) {
                Span[] spans = finder.find(tokens);
                for (Span span : spans) {
                    String entityText = String.join(" ", 
                        Arrays.copyOfRange(tokens, span.getStart(), span.getEnd()));
                    entities.add(new NamedEntity(entityText, span.getType(), 
                        span.getStart(), span.getEnd()));
                }
                finder.clearAdaptiveData();
            }

            return new AnalysisResult(sentences, tokens, posTags, entities);
        }

        @Override
        public void close() {
            for (NameFinderME finder : nameFinders) {
                finder.clearAdaptiveData();
            }
        }
    }

    static class AnalysisResult {
        final String[] sentences;
        final String[] tokens;
        final String[] posTags;
        final List<NamedEntity> entities;

        AnalysisResult(String[] sentences, String[] tokens, String[] posTags, List<NamedEntity> entities) {
            this.sentences = sentences;
            this.tokens = tokens;
            this.posTags = posTags;
            this.entities = entities;
        }

        @Override
        public String toString() {
            return String.format("Sentences: %d, Tokens: %d, Entities: %d",
                    sentences.length, tokens.length, entities.size());
        }
    }

    static class NamedEntity {
        final String text;
        final String type;
        final int start;
        final int end;

        NamedEntity(String text, String type, int start, int end) {
            this.text = text;
            this.type = type;
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return String.format("%s (%s)", text, type);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Text Classifier using OpenNLP Features
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Text classifier using OpenNLP-extracted features with Gollek neural network.
     */
    static class OpenNLPTextClassifier implements AutoCloseable {
        private final OpenNLPPipeline pipeline;
        private final tech.kayys.gollek.ml.nn.Module classifier;
        private final int featureDim;

        OpenNLPTextClassifier(int featureDim) throws IOException {
            this.featureDim = featureDim;
            this.pipeline = new OpenNLPPipeline();
            
            this.classifier = new Sequential(
                new Linear(featureDim, 32),
                new ReLU(),
                new Linear(32, 16),
                new ReLU(),
                new Linear(16, 3) // updated to 3 categories
            );

            System.out.println("📦 OpenNLP Text Classifier");
            System.out.printf("   Input: %d OpenNLP features%n", featureDim);
            System.out.printf("   Architecture: Linear(%d→32) → ReLU → Linear(32→16) → ReLU → Linear(16→3)%n", featureDim);
        }

        void train(List<String> texts, List<Integer> labels, int epochs, int batchSize) {
            System.out.println("\n🏋️ Training classifier...");

            // Extract features
            System.out.println("📊 Extracting features with OpenNLP...");
            float[][] features = new float[texts.size()][featureDim];
            for (int i = 0; i < texts.size(); i++) {
                features[i] = pipeline.extractFeatures(texts.get(i));
            }

            // Convert to tensors
            float[] featureData = new float[features.length * featureDim];
            for (int i = 0; i < features.length; i++) {
                System.arraycopy(features[i], 0, featureData, i * featureDim, featureDim);
            }
            float[] labelData = new float[labels.size()];
            for (int i = 0; i < labels.size(); i++) {
                labelData[i] = labels.get(i);
            }
            GradTensor inputTensor = GradTensor.of(featureData, features.length, featureDim);
            GradTensor labelTensor = GradTensor.of(labelData, labels.size());

            // Train
            var trainer = Trainer.builder()
                    .model(classifier)
                    .optimizer(new Adam(classifier.parameters(), 0.001f))
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

        Prediction predict(String text) {
            float[] features = pipeline.extractFeatures(text);
            GradTensor input = GradTensor.of(features, 1, featureDim);
            
            classifier.eval();
            GradTensor output = classifier.forward(input);
            float[] scores = output.data();

            int predictedClass = 0;
            float maxScore = scores[0];
            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i];
                    predictedClass = i;
                }
            }

            // Softmax
            float sum = 0f;
            float[] probs = new float[scores.length];
            for (int i = 0; i < scores.length; i++) {
                probs[i] = (float) Math.exp(scores[i] - maxScore);
                sum += probs[i];
            }
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }

            return new Prediction(predictedClass, probs, scores);
        }

        @Override
        public void close() {
            pipeline.close();
        }
    }

    static class Prediction {
        final int predictedClass;
        final float[] probabilities;
        final float[] rawScores;

        Prediction(int predictedClass, float[] probabilities, float[] rawScores) {
            this.predictedClass = predictedClass;
            this.probabilities = probabilities;
            this.rawScores = rawScores;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("Class=%d, Prob=[", predictedClass));
            for (int i = 0; i < probabilities.length; i++) {
                sb.append(String.format("%.3f", probabilities[i]));
                if (i < probabilities.length - 1) sb.append(", ");
            }
            float maxProb = 0f;
            for (float p : probabilities) if (p > maxProb) maxProb = p;
            sb.append(String.format("], Confidence=%.1f%%", maxProb * 100));
            return sb.toString();
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // NER-based Document Classifier
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Classifies documents based on named entity distribution.
     * Useful for categorizing news articles, reports, etc.
     */
    static class NERBasedClassifier implements AutoCloseable {
        private final OpenNLPPipeline pipeline;
        private final tech.kayys.gollek.ml.nn.Module classifier;
        private final int featureDim;

        NERBasedClassifier() throws IOException {
            this.pipeline = new OpenNLPPipeline();
            // Features: [person_ratio, location_ratio, org_ratio, total_ne_density]
            this.featureDim = 4;
            
            this.classifier = new Sequential(
                new Linear(featureDim, 16),
                new ReLU(),
                new Linear(16, 3) // 3 categories: News, Biography, General
            );

            System.out.println("📦 NER-Based Document Classifier");
            System.out.printf("   Features: %d (NER distribution)%n", featureDim);
        }

        float[] extractNERFeatures(String text) {
            float[] features = new float[featureDim];
            
            AnalysisResult result = pipeline.analyze(text);
            int totalTokens = result.tokens.length;

            int personCount = 0, locationCount = 0, orgCount = 0;
            for (NamedEntity entity : result.entities) {
                switch (entity.type) {
                    case "person" -> personCount++;
                    case "location" -> locationCount++;
                    case "organization" -> orgCount++;
                }
            }

            features[0] = (float) personCount / Math.max(1, totalTokens);
            features[1] = (float) locationCount / Math.max(1, totalTokens);
            features[2] = (float) orgCount / Math.max(1, totalTokens);
            features[3] = (float) result.entities.size() / Math.max(1, totalTokens);

            return features;
        }

        String classify(String text) {
            float[] features = extractNERFeatures(text);
            GradTensor input = GradTensor.of(features, 1, featureDim);
            
            classifier.eval();
            GradTensor output = classifier.forward(input);
            float[] scores = output.data();

            int maxIdx = 0;
            float maxScore = scores[0];
            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i];
                    maxIdx = i;
                }
            }

            String[] categories = {"News Article", "Biography", "General"};
            return categories[maxIdx];
        }

        @Override
        public void close() {
            pipeline.close();
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Demo Dataset
    // ──────────────────────────────────────────────────────────────────────

    static class DemoDataset {
        static List<String> getTrainingTexts() {
            List<String> texts = new ArrayList<>();
            
            // News-like texts (more organizations, locations)
            texts.add("The company announced quarterly earnings exceeding expectations");
            texts.add("Government officials met in Washington to discuss policy changes");
            texts.add("The stock market showed gains following the Federal Reserve announcement");
            texts.add("International leaders gathered at the United Nations summit");
            texts.add("The corporation reported record profits in the Asian market");
            
            // Biography-like texts (more persons)
            texts.add("John was born in a small town and grew up with his family");
            texts.add("She dedicated her life to helping others and inspiring change");
            texts.add("The artist created masterpieces throughout his remarkable career");
            texts.add("He served in the military before becoming a successful entrepreneur");
            texts.add("The scientist made groundbreaking discoveries in her field");
            
            // General texts
            texts.add("The weather forecast predicts rain for the weekend");
            texts.add("Cooking requires patience and attention to detail");
            texts.add("Technology continues to evolve at a rapid pace");
            texts.add("Education is fundamental to personal and societal growth");
            texts.add("Sports bring people together from all walks of life");

            return texts;
        }

        static List<Integer> getTrainingLabels() {
            // 0 = News, 1 = Biography, 2 = General
            List<Integer> labels = new ArrayList<>();
            for (int i = 0; i < 5; i++) labels.add(0);
            for (int i = 0; i < 5; i++) labels.add(1);
            for (int i = 0; i < 5; i++) labels.add(2);
            return labels;
        }

        static List<String> getTestTexts() {
            return Arrays.asList(
                "The president visited the headquarters to meet with executives",
                "Mary studied hard and graduated with honors from university",
                "The new software update includes several bug fixes"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main: Run All Integration Examples
    // ──────────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek SDK + Apache OpenNLP Integration Examples      ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        try {
            // Example 1: OpenNLP Pipeline Demo
            System.out.println("\n" + "═".repeat(60));
            System.out.println("EXAMPLE 1: OpenNLP Pipeline Analysis");
            System.out.println("═".repeat(60));
            
            try (var pipeline = new OpenNLPPipeline()) {
                String sampleText = "Apple Inc. was founded by Steve Jobs in California. " +
                                   "The company is now led by Tim Cook. " +
                                   "Apple headquarters is located in Cupertino!";

                System.out.printf("%n📝 Analyzing: \"%s\"%n", sampleText);
                
                AnalysisResult result = pipeline.analyze(sampleText);
                System.out.printf("%n📊 Analysis Results:%n");
                System.out.printf("   Sentences: %d%n", result.sentences.length);
                System.out.printf("   Tokens: %d%n", result.tokens.length);
                System.out.printf("   Named Entities: %d%n", result.entities.size());
                
                if (!result.entities.isEmpty()) {
                    System.out.println("\n🏷️ Named Entities:");
                    for (NamedEntity entity : result.entities) {
                        System.out.printf("   - %s (%s)%n", entity.text, entity.type);
                    }
                }

                System.out.println("\n📊 POS Tags (first 10 tokens):");
                for (int i = 0; i < Math.min(10, result.tokens.length); i++) {
                    System.out.printf("   %-15s → %s%n", result.tokens[i], result.posTags[i]);
                }

                // Feature extraction
                float[] features = pipeline.extractFeatures(sampleText);
                System.out.printf("%n📦 Extracted Features: %d dimensions%n", features.length);
                System.out.printf("   First 5 features: %s%n", 
                    Arrays.toString(Arrays.copyOfRange(features, 0, 5)));
            }

            // Example 2: Text Classification with OpenNLP Features
            System.out.println("\n" + "═".repeat(60));
            System.out.println("EXAMPLE 2: Text Classification with OpenNLP Features");
            System.out.println("═".repeat(60));
            
            int featureDim = 15;
            try (var classifier = new OpenNLPTextClassifier(featureDim)) {
                List<String> trainTexts = DemoDataset.getTrainingTexts();
                List<Integer> trainLabels = DemoDataset.getTrainingLabels();
                List<String> testTexts = DemoDataset.getTestTexts();

                classifier.train(trainTexts, trainLabels, 50, 8);

                System.out.println("\n📈 Testing predictions:");
                for (String text : testTexts) {
                    Prediction pred = classifier.predict(text);
                    System.out.printf("   Text: \"%s\"%n", text.length() > 50 ? text.substring(0, 50) + "..." : text);
                    System.out.printf("   → %s%n", pred);
                }
            }

            // Example 3: NER-Based Document Classification
            System.out.println("\n" + "═".repeat(60));
            System.out.println("EXAMPLE 3: NER-Based Document Classification");
            System.out.println("═".repeat(60));
            
            try (var nerClassifier = new NERBasedClassifier()) {
                List<String> testTexts = Arrays.asList(
                    "The CEO of Microsoft Satya Nadella announced new AI initiatives at the company headquarters in Redmond",
                    "Marie Curie was a pioneering physicist who conducted groundbreaking research on radioactivity in Paris",
                    "The recipe requires flour sugar eggs and butter mixed together and baked at 350 degrees"
                );

                System.out.println("\n📈 Classifying documents by NER distribution:");
                for (String text : testTexts) {
                    String category = nerClassifier.classify(text);
                    System.out.printf("   Text: \"%s\"%n", text.length() > 60 ? text.substring(0, 60) + "..." : text);
                    System.out.printf("   → Category: %s%n", category);
                }
            }

            System.out.println("\n╔══════════════════════════════════════════════════════════╗");
            System.out.println("║   ✅ All OpenNLP integration examples completed!        ║");
            System.out.println("╚══════════════════════════════════════════════════════════╝");

        } catch (Exception e) {
            System.err.println("❌ Error during integration demo: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
