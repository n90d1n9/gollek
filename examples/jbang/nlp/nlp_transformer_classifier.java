//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS mavenLocal,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

// TIP: Enable hardware acceleration on macOS (Apple Silicon) by adding this dependency:
//DEPS tech.kayys.gollek:gollek-plugin-metal:0.1.0-SNAPSHOT

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.Module;
import tech.kayys.gollek.spi.tensor.ComputeBackendRegistry;
import tech.kayys.gollek.ml.nn.loss.BCEWithLogitsLoss;
import tech.kayys.gollek.ml.nn.optim.Adam;
import tech.kayys.gollek.ml.nn.optim.SGD;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Professional NLP Example: Transformer-based Sentiment Classifier (Mini-BERT).
 * 
 * Demonstrates:
 * 1. Word-level tokenization and vocabulary management.
 * 2. Building a Transformer architecture (Embedding + Positional encoding + Encoder).
 * 3. Training on a synthetic sequence dataset for sentiment classification.
 * 4. Saving and reloading the model via Safetensors.
 */
public class nlp_transformer_classifier {

    // --- MINI-BERT ARCHITECTURE ---
    static class MiniBERT extends Module {
        private final Embedding embedding;
        private final PositionalEncoding posEncoding;
        private final TransformerEncoderLayer transformer;
        private final Linear classifier;
        private final int dModel;

        public MiniBERT(int vocabSize, int dModel, int nHeads, int maxLen) {
            this.dModel = dModel;
            this.embedding = register("embeddings", new Embedding(vocabSize, dModel));
            this.posEncoding = register("pos_encoding", new PositionalEncoding(dModel, maxLen, 0.1f));
            this.transformer = register("transformer", new TransformerEncoderLayer(dModel, nHeads, dModel * 4, 0.1f));
            this.classifier = register("classifier", new Linear(dModel, 1));
        }

        @Override
        public GradTensor forward(GradTensor input) {
            // 1. Embedding [batch, seq, dModel]
            GradTensor x = embedding.forward(input);
            // 2. Add Position info [batch, seq, dModel]
            x = posEncoding.forward(x);
            // 3. Transformer Block [batch, seq, dModel]
            x = transformer.forward(x);
            // 4. Global Mean Pooling [batch, dModel]
            x = globalMeanPool(x);
            // 5. Classification logic [batch, 1]
            return classifier.forward(x);
        }

        private GradTensor globalMeanPool(GradTensor x) {
            long[] shape = x.shape();
            int batch = (int) shape[0];
            int seqLen = (int) shape[1];
            int dim = (int) shape[2];

            float[] data = x.data();
            float[] pooled = new float[batch * dim];
            for (int b = 0; b < batch; b++) {
                for (int d = 0; d < dim; d++) {
                    float sum = 0;
                    for (int s = 0; s < seqLen; s++) {
                        sum += data[b * seqLen * dim + s * dim + d];
                    }
                    pooled[b * dim + d] = sum / seqLen;
                }
            }
            return GradTensor.of(pooled, batch, dim);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Gollek Transformer NLP Sentiment Analysis ===");

        // --- 1. PREPARE DATA ---
        String[][] corpus = {
            {"this", "system", "is", "amazing"}, {"positive"},
            {"i", "really", "love", "this", "tool"}, {"positive"},
            {"great", "performance", "and", "speed"}, {"positive"},
            {"this", "code", "is", "clean"}, {"positive"},
            
            {"this", "is", "really", "bad"}, {"negative"},
            {"i", "hate", "this", "buggy", "tool"}, {"negative"},
            {"slow", "system", "is", "annoying"}, {"negative"},
            {"errors", "and", "crashes", "everywhere"}, {"negative"}
        };

        // Build Vocabulary
        Set<String> vocab = new HashSet<>();
        vocab.add("[PAD]"); // ID 0 is reserved for padding
        for (int i = 0; i < corpus.length; i += 2) {
            vocab.addAll(Arrays.asList(corpus[i]));
        }
        List<String> vocabList = new ArrayList<>(vocab);
        Collections.sort(vocabList);
        Map<String, Integer> wordToId = new HashMap<>();
        for (int i = 0; i < vocabList.size(); i++) wordToId.put(vocabList.get(i), i);
        int vocabSize = vocabList.size();
        System.out.println("Vocabulary size: " + vocabSize);

        // Vectorize data (Max Sequence Length = 5)
        int maxLen = 5;
        float[] inputs = new float[(corpus.length / 2) * maxLen];
        float[] labels = new float[corpus.length / 2];

        for (int i = 0; i < corpus.length; i += 2) {
            String[] sentence = corpus[i];
            String sentiment = corpus[i+1][0];
            int row = i / 2;
            labels[row] = sentiment.equals("positive") ? 1.0f : 0.0f;
            for (int j = 0; j < maxLen; j++) {
                if (j < sentence.length) {
                    inputs[row * maxLen + j] = wordToId.getOrDefault(sentence[j], 0);
                } else {
                    inputs[row * maxLen + j] = 0; // Pad
                }
            }
        }

        GradTensor X = GradTensor.of(inputs, corpus.length / 2, maxLen);
        GradTensor Y = GradTensor.of(labels, corpus.length / 2, 1);

        // --- 2. TRAIN MINI-BERT ---
        int dModel = 32;
        MiniBERT model = new MiniBERT(vocabSize, dModel, 4, maxLen);
        
        System.out.println("Compute Device: " + ComputeBackendRegistry.get().deviceName());
        System.out.println("Model Params: " + model.parameterCountFormatted());

        BCEWithLogitsLoss criterion = new BCEWithLogitsLoss();
        Adam optimizer = new Adam(model.parameters(), 1e-3f); // Adam is much better for Transformers

        System.out.println("\nTraining Mini-BERT Classifier...");
        long start = System.currentTimeMillis();
        for (int i = 1; i <= 200; i++) {
            GradTensor output = model.forward(X);
            GradTensor loss = criterion.compute(output, Y);
            model.zeroGrad();
            loss.backward();
            optimizer.step();
            if (i % 50 == 0) System.out.printf("Iteration %d, Loss: %.4f\n", i, loss.item());
        }
        System.out.println("Training completed in " + (System.currentTimeMillis() - start) + "ms");

        // --- 3. PERSISTENCE (SAVE / LOAD) ---
        String filename = "mini_bert_sentiment.safetensors";
        System.out.println("\nSaving model to " + filename + "...");
        model.saveSafetensors(filename);

        MiniBERT loadedModel = new MiniBERT(vocabSize, dModel, 4, maxLen);
        loadedModel.loadSafetensors(filename);
        System.out.println("Model reloaded successfully!");

        // --- 4. INFERENCE ---
        System.out.println("\nTesting professional-grade inference:");
        testSentiment(loadedModel, "i love this coding tool", wordToId, maxLen);
        testSentiment(loadedModel, "system crashes and error", wordToId, maxLen);

        java.nio.file.Files.deleteIfExists(java.nio.file.Path.of(filename));
    }

    private static void testSentiment(MiniBERT model, String sentence, Map<String, Integer> wordToId, int maxLen) {
        String[] tokens = sentence.toLowerCase().split("\\s+");
        float[] input = new float[maxLen];
        for (int i = 0; i < maxLen; i++) {
            if (i < tokens.length) {
                input[i] = wordToId.getOrDefault(tokens[i], 0);
            } else {
                input[i] = 0;
            }
        }
        GradTensor out = model.forward(GradTensor.of(input, 1, maxLen));
        float prob = (float) (1.0 / (1.0 + Math.exp(-out.item()))); // Manual sigmoid for probability
        System.out.printf("Sentence: \"%s\" -> Sentiment: %s (Prob: %.2f%%)\n", 
            sentence, (prob > 0.5 ? "POSITIVE" : "NEGATIVE"), prob * 100);
    }
}
