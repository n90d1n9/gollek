//DEPS edu.stanford.nlp:stanford-corenlp:4.5.7
//DEPS edu.stanford.nlp:stanford-corenlp:4.5.7:models
//DEPS edu.stanford.nlp:stanford-corenlp:4.5.7:models-english
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-nn-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-autograd-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-runtime-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-api-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-model-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/jackson-databind-2.15.2.jar
//DEPS ${user.home}/.gollek/jbang/libs/jackson-core-2.15.2.jar
//DEPS ${user.home}/.gollek/jbang/libs/mutiny-3.1.1.jar
//DEPS com.google.code.gson:gson:2.11.0
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.simple.*;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.*;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss;
import tech.kayys.gollek.ml.nn.optim.Adam;

import java.util.*;
import java.util.stream.*;

/**
 * Stanford NLP Integration with Gollek SDK.
 *
 * This example demonstrates how to integrate Gollek SDK with Stanford CoreNLP,
 * a leading natural language processing toolkit. The integration leverages
 * Stanford's linguistic analysis capabilities with Gollek's neural networks.
 *
 * Use Cases:
 * - Using Stanford NLP for text preprocessing (tokenization, POS tagging, NER)
 * - Feeding linguistically-rich features into Gollek models
 * - Combining Stanford sentiment analysis with Gollek predictions
 * - Building multi-stage NLP pipelines
 * - Custom text classification with linguistic features
 *
 * Run: jbang stanford_nlp_integration.java
 */
public class stanford_nlp_integration {

    // ──────────────────────────────────────────────────────────────────────
    // Feature Extraction from Stanford NLP
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Extract rich linguistic features from text using Stanford NLP.
     * These features can be used as input to Gollek models.
     */
    static class LinguisticFeatureExtractor {
        private final StanfordCoreNLP pipeline;
        private final Map<String, Integer> featureIndex;
        private final int featureDim;

        /**
         * Create a linguistic feature extractor.
         * Features include:
         * - Sentence length
         * - Average word length
         * - POS tag distribution
         * - Named entity counts
         * - Sentiment score (from Stanford)
         * - Parse tree depth
         */
        LinguisticFeatureExtractor() {
            // Configure Stanford NLP pipeline
            Properties props = new Properties();
            props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,sentiment");
            props.setProperty("parse.maxlen", "70"); // Limit parse tree size
            this.pipeline = new StanfordCoreNLP(props);
            this.featureIndex = new HashMap<>();

            // Define feature dimensions
            // 1. Sentence length (1)
            // 2. Avg word length (1)
            // 3. POS tag distribution (12 tags)
            // 4. NER counts (7 categories)
            // 5. Stanford sentiment (5 classes)
            // 6. Parse tree depth (1)
            // 7. Punctuation count (1)
            // 8. Capitalized words ratio (1)
            this.featureDim = 1 + 1 + 12 + 7 + 5 + 1 + 1 + 1;

            System.out.println("📦 Linguistic Feature Extractor initialized");
            System.out.printf("   Feature dimension: %d%n", featureDim);
        }

        /**
         * Extract features from a single text.
         */
        float[] extractFeatures(String text) {
            float[] features = new float[featureDim];
            int idx = 0;

            // Create CoreNLP document
            CoreDocument doc = new CoreDocument(text);
            pipeline.annotate(doc);

            List<CoreLabel> tokens = doc.tokens();
            if (tokens.isEmpty()) {
                return features; // Return zero vector for empty text
            }

            // 1. Sentence length (normalized)
            features[idx++] = Math.min(1.0f, tokens.size() / 100.0f);

            // 2. Average word length (normalized)
            double avgWordLen = tokens.stream()
                    .mapToInt(t -> t.originalText().length())
                    .average()
                    .orElse(0.0);
            features[idx++] = (float) Math.min(1.0, avgWordLen / 15.0);

            // 3. POS tag distribution (12 common tags)
            Map<String, Integer> posCounts = new HashMap<>();
            for (CoreLabel token : tokens) {
                String pos = token.tag();
                posCounts.merge(pos, 1, Integer::sum);
            }
            String[] commonPOSTags = {"NN", "VB", "JJ", "RB", "PRP", "DT", "IN", "CC", "MD", "TO", "UH", "."};
            for (String posTag : commonPOSTags) {
                int count = posCounts.entrySet().stream()
                        .filter(e -> e.getKey().startsWith(posTag))
                        .mapToInt(Map.Entry::getValue)
                        .sum();
                features[idx++] = (float) count / tokens.size();
            }

            // 4. NER counts (7 categories)
            Map<String, Integer> nerCounts = new HashMap<>();
            for (CoreLabel token : tokens) {
                String ner = token.ner();
                if (!"O".equals(ner)) {
                    nerCounts.merge(ner, 1, Integer::sum);
                }
            }
            String[] nerCategories = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"};
            for (String ner : nerCategories) {
                features[idx++] = nerCounts.getOrDefault(ner, 0) / (float) tokens.size();
            }

            // 5. Stanford sentiment (5 classes)
            // Get overall document sentiment
            String sentiment = doc.sentences().stream()
                    .map(s -> s.sentiment())
                    .filter(Objects::nonNull)
                    .findFirst()
                    .orElse("Neutral");
            
            String[] sentiments = {"Very negative", "Negative", "Neutral", "Positive", "Very positive"};
            for (int i = 0; i < sentiments.length; i++) {
                features[idx++] = sentiment.equals(sentiments[i]) ? 1.0f : 0.0f;
            }

            // 6. Parse tree depth (normalized)
            int maxDepth = doc.sentences().stream()
                    .map(s -> s.coreMap().get(TreeCoreAnnotations.TreeAnnotation.class))
                    .filter(Objects::nonNull)
                    .mapToInt(t -> calculateTreeDepth((Tree)t, 0))
                    .max()
                    .orElse(0);
            features[idx++] = Math.min(1.0f, maxDepth / 20.0f);

            // 7. Punctuation count ratio
            long punctCount = tokens.stream()
                    .map(t -> t.originalText())
                    .filter(w -> w.matches("[^\\w\\s]"))
                    .count();
            features[idx++] = (float) punctCount / tokens.size();

            // 8. Capitalized words ratio
            long capitalizedCount = tokens.stream()
                    .map(t -> t.originalText())
                    .filter(w -> !w.isEmpty() && Character.isUpperCase(w.charAt(0)))
                    .count();
            features[idx++] = (float) capitalizedCount / tokens.size();

            return features;
        }

        private int calculateTreeDepth(edu.stanford.nlp.trees.Tree tree, int depth) {
            if (tree.isLeaf()) {
                return depth;
            }
            int maxChildDepth = 0;
            for (Tree child : tree.children()) {
                maxChildDepth = Math.max(maxChildDepth, calculateTreeDepth(child, depth + 1));
            }
            return maxChildDepth;
        }

        /**
         * Extract features from multiple texts (batch processing).
         */
        float[][] extractFeaturesBatch(List<String> texts) {
            float[][] allFeatures = new float[texts.size()][featureDim];
            for (int i = 0; i < texts.size(); i++) {
                allFeatures[i] = extractFeatures(texts.get(i));
            }
            return allFeatures;
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Text Classification Model using Linguistic Features
    // ──────────────────────────────────────────────────────────────────────

    /**
     * A text classifier that uses Stanford NLP features with Gollek neural network.
     */
    static class LinguisticTextClassifier {
        private final LinguisticFeatureExtractor featureExtractor;
        private final tech.kayys.gollek.ml.nn.Module classifier;
        private final int featureDim;

        LinguisticTextClassifier(int featureDim) {
            this.featureDim = featureDim;
            this.featureExtractor = new LinguisticFeatureExtractor();
            
            // Build classifier: Feature Vector → Hidden → Output
            this.classifier = new Sequential(
                new Linear(featureDim, 64),
                new ReLU(),
                new Dropout(0.3f),
                new Linear(64, 32),
                new ReLU(),
                new Linear(32, 2) // Binary classification
            );

            System.out.println("📦 Linguistic Text Classifier");
            System.out.printf("   Input: %d linguistic features%n", featureDim);
            System.out.printf("   Architecture: Linear(%d→64) → ReLU → Dropout → Linear(64→32) → ReLU → Linear(32→2)%n", featureDim);
        }

        /**
         * Train the classifier on labeled texts.
         */
        void train(List<String> texts, List<Integer> labels, int epochs, int batchSize) {
            System.out.println("\n🏋️ Training on linguistic features...");

            // Extract features using Stanford NLP
            System.out.println("📊 Extracting linguistic features with Stanford CoreNLP...");
            float[][] features = featureExtractor.extractFeaturesBatch(texts);
            
            // Convert to tensors
            float[] featureData = new float[features.length * featureDim];
            for (int i = 0; i < features.length; i++) {
                System.arraycopy(features[i], 0, featureData, i * featureDim, featureDim);
            }
            GradTensor inputTensor = GradTensor.of(featureData, features.length, featureDim);
            float[] labelArray = new float[labels.size()];
            for (int i = 0; i < labels.size(); i++) {
                labelArray[i] = labels.get(i);
            }
            GradTensor labelTensor = GradTensor.of(labelArray, labels.size());

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

        /**
         * Predict class for a single text.
         */
        Prediction predict(String text) {
            float[] features = featureExtractor.extractFeatures(text);
            GradTensor input = GradTensor.of(features, 1, featureDim);
            
            classifier.eval();
            GradTensor output = classifier.forward(input);
            float[] scores = output.data();

            int predictedClass = scores[0] > scores[1] ? 0 : 1;
            
            // Apply softmax for probability
            float maxScore = Math.max(scores[0], scores[1]);
            float exp0 = (float) Math.exp(scores[0] - maxScore);
            float exp1 = (float) Math.exp(scores[1] - maxScore);
            float sum = exp0 + exp1;
            float prob0 = exp0 / sum;
            float prob1 = exp1 / sum;

            return new Prediction(predictedClass, new float[]{prob0, prob1}, scores);
        }

        /**
         * Predict classes for multiple texts.
         */
        List<Prediction> predictBatch(List<String> texts) {
            return texts.stream().map(this::predict).collect(Collectors.toList());
        }


    }

    // ──────────────────────────────────────────────────────────────────────
    // Prediction Result Class
    // ──────────────────────────────────────────────────────────────────────

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
            return String.format("Class=%d, Prob=[%.3f, %.3f], Confidence=%.1f%%",
                    predictedClass, probabilities[0], probabilities[1],
                    Math.max(probabilities[0], probabilities[1]) * 100);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Integration Pattern: Stanford Sentiment + Gollek Ensemble
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Combines Stanford's built-in sentiment analysis with Gollek predictions.
     * This ensemble approach can provide more robust sentiment classification.
     */
    static class HybridSentimentAnalyzer {
        private final edu.stanford.nlp.pipeline.StanfordCoreNLP pipeline;
        private final tech.kayys.gollek.ml.nn.Module gollekClassifier;
        private final LinguisticFeatureExtractor featureExtractor;
        private final int featureDim;

        HybridSentimentAnalyzer(int featureDim) {
            this.featureDim = featureDim;
            this.featureExtractor = new LinguisticFeatureExtractor();
            this.pipeline = new StanfordCoreNLP(new Properties() {{
                setProperty("annotators", "tokenize,ssplit,pos,parse,sentiment");
            }});
            this.gollekClassifier = new Sequential(
                new Linear(featureDim, 32),
                new ReLU(),
                new Linear(32, 2)
            );
        }

        /**
         * Analyze sentiment using both Stanford and Gollek, then combine.
         */
        SentimentResult analyze(String text) {
            // Stanford NLP sentiment (built-in)
            CoreDocument doc = new CoreDocument(text);
            pipeline.annotate(doc);

            String stanfordSentiment = doc.sentences().stream()
                    .map(s -> s.sentiment())
                    .filter(Objects::nonNull)
                    .findFirst()
                    .orElse("Neutral");

            // Convert Stanford sentiment to binary (negative/positive)
            int stanfordClass = stanfordSentiment.contains("negative") || 
                               stanfordSentiment.equals("Neutral") ? 0 : 1;

            // Gollek prediction
            float[] features = featureExtractor.extractFeatures(text);
            GradTensor input = GradTensor.of(features, 1, featureDim);
            gollekClassifier.eval();
            GradTensor output = gollekClassifier.forward(input);
            float[] gollekScores = output.data();
            int gollekClass = gollekScores[0] > gollekScores[1] ? 0 : 1;

            // Ensemble: simple voting
            int ensembleClass = (stanfordClass + gollekClass) >= 1 ? 1 : 0;

            return new SentimentResult(stanfordSentiment, stanfordClass, gollekClass, ensembleClass);
        }

        void train(List<String> texts, List<Integer> labels, int epochs) {
            float[][] features = featureExtractor.extractFeaturesBatch(texts);
            float[] featureData = new float[features.length * featureDim];
            for (int i = 0; i < features.length; i++) {
                System.arraycopy(features[i], 0, featureData, i * featureDim, featureDim);
            }
            GradTensor inputTensor = GradTensor.of(featureData, features.length, featureDim);
            float[] labelArray = new float[labels.size()];
            for (int i = 0; i < labels.size(); i++) {
                labelArray[i] = labels.get(i);
            }
            GradTensor labelTensor = GradTensor.of(labelArray, labels.size());

            var trainer = Trainer.builder()
                    .model(gollekClassifier)
                    .optimizer(new Adam(gollekClassifier.parameters(), 0.001f))
                    .lossFunction((pred, target) -> new CrossEntropyLoss().compute(pred, target))
                    .epochs(epochs)
                    .build();

            trainer.fit(inputTensor, labelTensor, 16);
        }


    }

    static class SentimentResult {
        final String stanfordSentiment;
        final int stanfordClass;
        final int gollekClass;
        final int ensembleClass;

        SentimentResult(String stanfordSentiment, int stanfordClass, int gollekClass, int ensembleClass) {
            this.stanfordSentiment = stanfordSentiment;
            this.stanfordClass = stanfordClass;
            this.gollekClass = gollekClass;
            this.ensembleClass = ensembleClass;
        }

        @Override
        public String toString() {
            return String.format("Stanford=%s(%d), Gollek=%d, Ensemble=%d",
                    stanfordSentiment, stanfordClass, gollekClass, ensembleClass);
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Demo Dataset
    // ──────────────────────────────────────────────────────────────────────

    static class TextDataset {
        static List<String> getTrainingTexts() {
            List<String> texts = new ArrayList<>();
            
            // Positive examples
            texts.add("This movie is absolutely fantastic and I loved every moment of it");
            texts.add("The acting was superb and the storyline kept me engaged throughout");
            texts.add("A masterpiece of modern cinema with brilliant performances");
            texts.add("Highly recommended for anyone who enjoys quality entertainment");
            texts.add("The director did an amazing job bringing this story to life");
            texts.add("Beautiful cinematography and excellent soundtrack");
            texts.add("One of the best films I have seen this year");
            texts.add("The characters were well developed and relatable");
            texts.add("A heartwarming story that will make you smile");
            texts.add("Outstanding performances from the entire cast");
            
            // Negative examples
            texts.add("This was a complete waste of time and money");
            texts.add("The plot made no sense and the acting was terrible");
            texts.add("I could not wait for this boring movie to end");
            texts.add("Avoid this film at all costs it is dreadful");
            texts.add("The worst movie I have seen in years");
            texts.add("Poorly written and poorly executed");
            texts.add("I regret watching this disappointing film");
            texts.add("The special effects were laughably bad");
            texts.add("A complete disaster from start to finish");
            texts.add("Do not bother watching this terrible movie");

            return texts;
        }

        static List<Integer> getTrainingLabels() {
            // 1 = positive, 0 = negative
            List<Integer> labels = new ArrayList<>();
            for (int i = 0; i < 10; i++) labels.add(1);
            for (int i = 0; i < 10; i++) labels.add(0);
            return labels;
        }

        static List<String> getTestTexts() {
            return Arrays.asList(
                "The film was good but could have been better",
                "Absolutely terrible experience I hated it",
                "A wonderful movie with great performances",
                "Not the best but still watchable",
                "I would not recommend this to anyone"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Main: Run All Integration Examples
    // ──────────────────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   Gollek SDK + Stanford NLP Integration Examples        ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        // Get dataset
        List<String> trainTexts = TextDataset.getTrainingTexts();
        List<Integer> trainLabels = TextDataset.getTrainingLabels();
        List<String> testTexts = TextDataset.getTestTexts();

        System.out.printf("%n📚 Dataset: %d training texts, %d test texts%n", 
                trainTexts.size(), testTexts.size());

        try {
            // Example 1: Linguistic Feature Extraction Demo
            System.out.println("\n" + "═".repeat(60));
            System.out.println("EXAMPLE 1: Linguistic Feature Extraction");
            System.out.println("═".repeat(60));
            
            var extractor = new LinguisticFeatureExtractor();
            System.out.printf("%n📝 Extracting features for: \"%s\"%n", "John works at Google in California. He loves his job!");
            
            float[] features = extractor.extractFeatures("John works at Google in California. He loves his job!");
            System.out.printf("   Feature dimension: %d%n", features.length);
            System.out.printf("   First 10 features: %s%n", 
                Arrays.toString(Arrays.copyOfRange(features, 0, Math.min(10, features.length))));

            // Example 2: Text Classification with Linguistic Features
            System.out.println("\n" + "═".repeat(60));
            System.out.println("EXAMPLE 2: Text Classification with Linguistic Features");
            System.out.println("═".repeat(60));
            
            int featureDim = 29; // As defined in LinguisticFeatureExtractor
            var classifier = new LinguisticTextClassifier(featureDim);
            classifier.train(trainTexts, trainLabels, 30, 8);

            System.out.println("\n📈 Testing predictions:");
            for (String text : testTexts) {
                Prediction pred = classifier.predict(text);
                System.out.printf("   Text: \"%s\"%n", text.length() > 50 ? text.substring(0, 50) + "..." : text);
                System.out.printf("   → %s%n", pred);
            }

            // Example 3: Hybrid Sentiment Analysis (Stanford + Gollek)
            System.out.println("\n" + "═".repeat(60));
            System.out.println("EXAMPLE 3: Hybrid Sentiment Analysis (Stanford + Gollek)");
            System.out.println("═".repeat(60));
            
            var hybridAnalyzer = new HybridSentimentAnalyzer(featureDim);
            // Train Gollek component
            hybridAnalyzer.train(trainTexts, trainLabels, 30);

            System.out.println("\n📈 Testing hybrid sentiment analysis:");
            for (String text : testTexts) {
                SentimentResult result = hybridAnalyzer.analyze(text);
                System.out.printf("   Text: \"%s\"%n", text.length() > 50 ? text.substring(0, 50) + "..." : text);
                System.out.printf("   → %s%n", result);
            }

            // Example 4: Detailed NLP Pipeline Demo
            System.out.println("\n" + "═".repeat(60));
            System.out.println("EXAMPLE 4: Stanford NLP Pipeline Details");
            System.out.println("═".repeat(60));
            
            Properties props = new Properties();
            props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
            
            String demoText = "Apple Inc. was founded by Steve Jobs in 1976. The company is headquartered in California.";
            CoreDocument doc = new CoreDocument(demoText);
            pipeline.annotate(doc);

            System.out.printf("%n📝 Analyzing: \"%s\"%n", demoText);
            System.out.println("\n📊 Tokens with POS and NER:");
            for (CoreLabel token : doc.tokens()) {
                System.out.printf("   %-15s POS=%-6s NER=%-15s Lemma=%s%n",
                        token.originalText(),
                        token.tag(),
                        token.ner(),
                        token.lemma());
            }

            System.out.println("\n╔══════════════════════════════════════════════════════════╗");
            System.out.println("║   ✅ All Stanford NLP integration examples completed!   ║");
            System.out.println("╚══════════════════════════════════════════════════════════╝");

        } catch (Exception e) {
            System.err.println("❌ Error during integration demo: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
