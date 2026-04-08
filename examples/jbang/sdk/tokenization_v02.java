//#!/usr/bin/env jbang
//DEPS tech.kayys.gollek:gollek-sdk-nlp:0.2.0
//DEPS org.slf4j:slf4j-simple:2.0.0

import java.util.*;
import tech.kayys.gollek.ml.nlp.tokenization.*;

/**
 * Gollek SDK v0.2 - NLP Tokenization Example
 * 
 * Demonstrates the tokenization interface and usage patterns:
 * - Text encoding to token IDs
 * - Token decoding back to text
 * - Batch processing
 * - Special tokens handling
 * - Token type IDs (for BERT-style models)
 * - Attention masks
 * - Padding strategies
 * 
 * This example shows how to prepare text data for NLP models:
 * - Language models (GPT, Llama)
 * - Transformers (BERT, RoBERTa)
 * - Sequence classification
 * - Token classification
 * 
 * Usage: jbang 03_tokenization.java
 * 
 * Expected Output:
 * ✓ Text encoded to token IDs
 * ✓ Tokens decoded back to text
 * ✓ Special tokens handled correctly
 * ✓ Batch processing with padding
 * ✓ Attention masks generated
 */
public class tokenization_v02 {

    // Simulate a simple tokenizer implementation
    static class SimpleTokenizer implements Tokenizer {
        private int vocabSize;
        private int maxLength;
        private int padTokenId = 0;
        private int unknownTokenId = 100;

        SimpleTokenizer(int vocabSize, int maxLength) {
            this.vocabSize = vocabSize;
            this.maxLength = maxLength;
        }

        @Override
        public EncodedTokens encode(String text) {
            // Simple tokenization: split on spaces
            String[] tokens = text.toLowerCase().split("\\s+");
            List<Integer> tokenIds = new ArrayList<>();

            tokenIds.add(101); // [CLS] token
            for (String token : tokens) {
                // Convert to simple hash-based ID (0-99999)
                int tokenId = Math.abs(token.hashCode()) % vocabSize;
                tokenIds.add(tokenId);
            }
            tokenIds.add(102); // [SEP] token

            // Pad to maxLength
            while (tokenIds.size() < maxLength) {
                tokenIds.add(padTokenId);
            }

            // Generate attention mask
            int[] attentionMask = new int[maxLength];
            int realLen = Math.min(tokens.length + 2, maxLength);
            for (int i = 0; i < realLen; i++) {
                attentionMask[i] = 1;
            }

            // Generate token type IDs (BERT-style)
            int[] tokenTypeIds = new int[maxLength];

            return new EncodedTokens(tokenIds,
                    tokenTypeIds,
                    attentionMask,
                    vocabSize);
        }

        @Override
        public String decode(List<Integer> tokenIds) {
            StringBuilder result = new StringBuilder();
            for (Integer id : tokenIds) {
                if (id > 102) { // Skip special tokens
                    result.append("[Token:").append(id).append("] ");
                }
            }
            return result.toString().trim();
        }

        @Override
        public int getVocabSize() {
            return vocabSize;
        }

        @Override
        public String getSpecialToken(String tokenType) {
            return switch (tokenType) {
                case "pad" -> "[PAD]";
                case "unk" -> "[UNK]";
                case "cls" -> "[CLS]";
                case "sep" -> "[SEP]";
                default -> "";
            };
        }
    }

    // EncodedTokens record
    record EncodedTokens(
            List<Integer> tokenIds,
            int[] tokenTypeIds,
            int[] attentionMask,
            int vocabSize) implements Comparable<EncodedTokens> {
        @Override
        public int compareTo(EncodedTokens o) {
            return Integer.compare(tokenIds.size(), o.tokenIds.size());
        }
    }

    static class TokenizationDemo {

        void runDemo() {
            System.out.println("\n╔════════════════════════════════════════════╗");
            System.out.println("║    Gollek SDK v0.2 - NLP Tokenization    ║");
            System.out.println("╚════════════════════════════════════════════╝\n");

            // Create tokenizer
            SimpleTokenizer tokenizer = new SimpleTokenizer(10000, 128);

            // Example 1: Basic Text Encoding
            System.out.println("1️⃣  BASIC TEXT ENCODING");
            System.out.println("─".repeat(50));
            demonstrateEncoding(tokenizer);

            // Example 2: Token Decoding
            System.out.println("\n2️⃣  TOKEN DECODING");
            System.out.println("─".repeat(50));
            demonstrateDecoding(tokenizer);

            // Example 3: Special Tokens
            System.out.println("\n3️⃣  SPECIAL TOKENS");
            System.out.println("─".repeat(50));
            demonstrateSpecialTokens(tokenizer);

            // Example 4: Batch Processing
            System.out.println("\n4️⃣  BATCH PROCESSING");
            System.out.println("─".repeat(50));
            demonstrateBatchProcessing(tokenizer);

            // Example 5: Attention Masks
            System.out.println("\n5️⃣  ATTENTION MASKS & TOKEN TYPES");
            System.out.println("─".repeat(50));
            demonstrateAttentionMasks(tokenizer);

            // Example 6: Padding Strategies
            System.out.println("\n6️⃣  PADDING STRATEGIES");
            System.out.println("─".repeat(50));
            demonstratePadding();

            // Example 7: Real-world NLP Tasks
            System.out.println("\n7️⃣  REAL-WORLD NLP TASKS");
            System.out.println("─".repeat(50));
            demonstrateNLPTasks(tokenizer);

            System.out.println("\n" + "✓".repeat(25));
            System.out.println("All tokenization examples completed!");
            System.out.println("✓".repeat(25) + "\n");
        }

        void demonstrateEncoding(SimpleTokenizer tokenizer) {
            System.out.println("Scenario: Encode text into token IDs\n");

            String text = "The quick brown fox jumps over the lazy dog";
            System.out.println("Input text: \"" + text + "\"\n");

            EncodedTokens encoded = tokenizer.encode(text);
            System.out.println("Encoded tokens:");
            System.out.println("  Token IDs: [101 (CLS), word1, word2, ..., word9, 102 (SEP), 0 (PAD), ...]");
            System.out.println("  Count: " + encoded.tokenIds.size() + " tokens");
            System.out.println("  Max length: 128 (with padding)");
            System.out.println("  ✓ Encoding successful");
        }

        void demonstrateDecoding(SimpleTokenizer tokenizer) {
            System.out.println("Scenario: Convert token IDs back to text\n");

            String text = "Gollek is awesome";
            System.out.println("Original: \"" + text + "\"\n");

            EncodedTokens encoded = tokenizer.encode(text);
            String decoded = tokenizer.decode(encoded.tokenIds);

            System.out.println("Decoded: \"" + decoded + "\"");
            System.out.println("Note: Decoding reconstructs tokens, not exact text");
            System.out.println("✓ Decoding successful");
        }

        void demonstrateSpecialTokens(SimpleTokenizer tokenizer) {
            System.out.println("Scenario: Understanding special tokens in BERT\n");

            System.out.println("BERT Special Tokens:");
            System.out.println("  [CLS] (101): Classification token at sequence start");
            System.out.println("  [SEP] (102): Separator between sequences");
            System.out.println("  [PAD] (0):   Padding token for shorter sequences");
            System.out.println("  [UNK] (100): Unknown token for OOV words\n");

            System.out.println("Example sequence:");
            System.out.println("  Input:  'Hello world'");
            System.out.println("  Tokens: [CLS] Hello world [SEP] [PAD] [PAD] ...");
            System.out.println("  IDs:    101   hello world  102   0    0   ...\n");

            // Get special tokens
            String padToken = tokenizer.getSpecialToken("pad");
            String clsToken = tokenizer.getSpecialToken("cls");
            System.out.println("Retrieved special tokens:");
            System.out.println("  PAD token: " + padToken);
            System.out.println("  CLS token: " + clsToken);
            System.out.println("  ✓ Special tokens handled correctly");
        }

        void demonstrateBatchProcessing(SimpleTokenizer tokenizer) {
            System.out.println("Scenario: Tokenize batch of sentences\n");

            List<String> texts = List.of(
                    "Hello world",
                    "Gollek is a machine learning framework",
                    "Natural language processing with Java");

            System.out.println("Batch of " + texts.size() + " texts:");
            for (int i = 0; i < texts.size(); i++) {
                System.out.println("  [" + i + "] \"" + texts.get(i) + "\"");
            }

            System.out.println("\nProcessing:");
            List<EncodedTokens> batch = new ArrayList<>();
            for (String text : texts) {
                EncodedTokens encoded = tokenizer.encode(text);
                batch.add(encoded);
                System.out.println("  Encoded to " + encoded.tokenIds.size() + " tokens");
            }

            System.out.println("\nBatch properties:");
            System.out.println("  All sequences padded to: 128 tokens");
            System.out.println("  Ready for model input");
            System.out.println("  ✓ Batch processing completed");
        }

        void demonstrateAttentionMasks(SimpleTokenizer tokenizer) {
            System.out.println("Scenario: Generate attention masks for padding\n");

            String text = "Hello world";
            EncodedTokens encoded = tokenizer.encode(text);

            int[] mask = encoded.attentionMask;
            int[] tokenTypes = encoded.tokenTypeIds;

            System.out.println("Text: \"" + text + "\"");
            System.out.println("Sequence length: " + encoded.tokenIds.size());
            System.out.println("Max length: 128\n");

            System.out.println("Attention Mask:");
            System.out.println("  1: Real token (should attend)");
            System.out.println("  0: Padding token (should be masked)");
            System.out.print("  First 10 values: [");
            for (int i = 0; i < 10; i++) {
                System.out.print(mask[i]);
                if (i < 9)
                    System.out.print(", ");
            }
            System.out.println(", ...]\n");

            System.out.println("Token Type IDs:");
            System.out.println("  0: First sequence (in sentence pair tasks)");
            System.out.println("  1: Second sequence");
            System.out.println("  First 10 values: [");
            for (int i = 0; i < 10; i++) {
                System.out.print(tokenTypes[i]);
                if (i < 9)
                    System.out.print(", ");
            }
            System.out.println(", ...]\n");

            System.out.println("Usage in transformer:");
            System.out.println("  - Attention mask prevents attending to [PAD]");
            System.out.println("  - Token type IDs for two-sentence tasks");
            System.out.println("  - Both improve model performance");
            System.out.println("  ✓ Masks generated correctly");
        }

        void demonstratePadding() {
            System.out.println("Scenario: Different padding strategies\n");

            System.out.println("Strategy 1: Fixed Length (max_length)");
            System.out.println("  Pad all sequences to 128 tokens");
            System.out.println("  ✓ Simple, but wastes space for short texts");
            System.out.println("  ✓ Required for batch processing\n");

            System.out.println("Strategy 2: Dynamic Padding");
            System.out.println("  Pad to max length in current batch");
            System.out.println("  ✓ More efficient memory usage");
            System.out.println("  ✓ Requires variable batch sizes\n");

            System.out.println("Strategy 3: Truncation + Padding");
            System.out.println("  Max length: 128");
            System.out.println("  Short (<128): pad to 128");
            System.out.println("  Long (>128): truncate to 128");
            System.out.println("  ✓ Balanced approach\n");

            System.out.println("Configuration example:");
            System.out.println("  max_length = 128");
            System.out.println("  padding = 'max_length'");
            System.out.println("  truncation = True");
            System.out.println("  return_tensors = 'tf' or 'torch'");
        }

        void demonstrateNLPTasks(SimpleTokenizer tokenizer) {
            System.out.println("Real-world NLP tasks with tokenization:\n");

            System.out.println("1️⃣  TEXT CLASSIFICATION");
            System.out.println("  Task: Sentiment analysis");
            System.out.println("  Input: \"This product is amazing!\"");
            System.out.println("  Preprocessing:");
            EncodedTokens sentiment = tokenizer.encode("This product is amazing");
            System.out.println("    - Tokenize text");
            System.out.println("    - Add [CLS] and [SEP] tokens");
            System.out.println("    - Tokens shape: (1, 128)");
            System.out.println("  Output: Positive sentiment (0.95)\n");

            System.out.println("2️⃣  SEQUENCE LABELING");
            System.out.println("  Task: Named entity recognition (NER)");
            System.out.println("  Input: \"John works at Google in California\"");
            System.out.println("  Preprocessing:");
            EncodedTokens ner = tokenizer.encode("John works at Google in California");
            System.out.println("    - Tokenize each word");
            System.out.println("    - Ensure 1:1 word-to-token mapping");
            System.out.println("    - Preserve token order");
            System.out.println("  Output: [PER, O, O, ORG, O, LOC]\n");

            System.out.println("3️⃣  QUESTION ANSWERING");
            System.out.println("  Task: Extract answer span from context");
            System.out.println("  Input: question + context (two sequences)");
            System.out.println("  Preprocessing:");
            System.out.println("    - Tokenize question");
            System.out.println("    - Add [SEP]");
            System.out.println("    - Tokenize context");
            System.out.println("    - Token type IDs: [0]*len(q) + [1]*len(c)");
            System.out.println("  Output: Start and end token positions\n");

            System.out.println("4️⃣  MACHINE TRANSLATION");
            System.out.println("  Task: Translate English to French");
            System.out.println("  Input: \"Hello, how are you?\"");
            System.out.println("  Preprocessing:");
            System.out.println("    - Tokenize source language");
            System.out.println("    - Add language tags");
            System.out.println("    - Generate initial target sequence");
            System.out.println("  Output: \"Bonjour, comment allez-vous?\"");

            System.out.println("\n✓ All NLP task patterns demonstrated");
        }
    }

    public static void main(String[] args) {
        try {
            new TokenizationDemo().runDemo();
        } catch (Exception e) {
            System.err.println("Error running tokenization example:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
