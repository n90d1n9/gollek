///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.gollek:gollek-ml-nlp:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-tensor:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-tokenizer-core:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector

import tech.kayys.gollek.ml.nlp.*;
import tech.kayys.gollek.ml.tensor.Tensor;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;

/**
 * Example demonstrating the High-Performance NLP Engine in Gollek.
 * 
 * Features showcased:
 * 1. Spacy-like pipeline (Doc, Token, Span)
 * 2. Vector API accelerated similarity (SIMD)
 * 3. Batched Embedding Inference
 * 4. Zero-copy Serialization with DocBin (FFM Memory Mapping)
 * 
 * Run with:
 * jbang examples/jbang/nlp/nlp_high_performance.java
 */
public class nlp_high_performance {

    public static void main(String[] args) throws Exception {
        System.out.println("=== Gollek High-Performance NLP Demo (JDK 25) ===\n");

        // 1. Initialize the Language Pipeline
        // In a real scenario, this loads a BPE tokenizer and an Embedding model
        Language nlp = LanguageFactory.load("en_core_web_sm");

        // 2. Process text into a Doc
        String text = "Gollek is a powerful ML framework optimized for Java 25 and modern hardware.";
        Doc doc = nlp.process(text);

        System.out.println("Processed Doc: " + doc);
        System.out.println("Tokens count: " + doc.length());

        // 3. Inspect Token attributes (Rule-based + Neural)
        System.out.println("\n--- Token Attributes ---");
        for (int i = 0; i < Math.min(5, doc.length()); i++) {
            Token token = doc.get(i);
            System.out.printf("Token[%d]: '%s' | isAlpha: %b\n",
                    token.getI(), token.getText(), token.isAlpha());
        }

        // 4. Vector API Accelerated Similarity (SIMD)
        // Let's create another doc and compare them
        Doc otherDoc = nlp.process("Modern hardware acceleration is key for ML performance.");

        // Simulating embedding vectors if the pipeline didn't populate them (for demo
        // purposes)
        if (doc.getVector() == null)
            doc.setVector(Tensor.randn(384));
        if (otherDoc.getVector() == null)
            otherDoc.setVector(Tensor.randn(384));

        double similarity = doc.similarity(otherDoc);
        System.out.printf("\nSemantic Similarity (SIMD-accelerated): %.4f\n", similarity);

        // 5. Batch Processing (nlp.pipe)
        System.out.println("\n--- Batch Processing ---");
        Stream<String> texts = Stream.of(
                "Java 25 brings FFM and Vector API.",
                "Spacy-like pipelines are intuitive.",
                "Gollek leverages hardware acceleration.");

        long count = nlp.pipe(texts).count();
        System.out.println("Processed batch of " + count + " documents.");

        // 6. Zero-Copy Serialization (DocBin)
        System.out.println("\n--- Zero-Copy Serialization (DocBin) ---");
        DocBin bin = new DocBin();
        bin.add(doc);
        bin.add(otherDoc);

        Path binPath = Paths.get("docs.bin");
        bin.toFile(binPath);
        System.out.println("Saved DocBin to: " + binPath.toAbsolutePath());

        // Load back using memory mapping (Zero-copy)
        DocBin loadedBin = DocBin.fromFile(binPath);
        System.out.println("Loaded Docs from bin: " + loadedBin.getDocs().size());

        // Cleanup
        binPath.toFile().delete();

        System.out.println("\nDemo completed successfully!");
    }
}
