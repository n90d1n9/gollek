///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.gollek:gollek-ml-nlp:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-tensor:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector

import tech.kayys.gollek.ml.nlp.*;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Multi-variant NLP Example: Concurrent Batch Processing.
 * 
 * This example demonstrates high-throughput text processing using
 * nlp.pipe() combined with Java's Parallel Streams.
 */
public class nlp_concurrent_batch_pipeline {

    public static void main(String[] args) {
        System.out.println("=== Gollek Concurrent NLP Pipeline Demo ===");

        // 1. Initialize Pipeline
        Language nlp = LanguageFactory.load("en_core_web_lg");

        // 2. Generate a large batch of synthetic data
        int batchSize = 1000;
        System.out.println("Generating " + batchSize + " synthetic documents...");
        List<String> texts = IntStream.range(0, batchSize)
                .mapToObj(i -> "This is document number " + i
                        + " containing some sample text for batch processing performance evaluation.")
                .toList();

        // 3. Process concurrently using Parallel Streams
        System.out.println("Processing batch concurrently...");
        long start = System.currentTimeMillis();

        long totalTokens = nlp.pipe(texts.parallelStream())
                .mapToLong(Doc::length)
                .sum();

        long end = System.currentTimeMillis();

        // 4. Report Performance
        double durationSeconds = (end - start) / 1000.0;
        double docsPerSecond = batchSize / durationSeconds;

        System.out.printf("\nSuccess! Processed %d documents in %.2f seconds.\n", batchSize, durationSeconds);
        System.out.printf("Throughput: %.2f docs/sec\n", docsPerSecond);
        System.out.printf("Total tokens indexed: %d\n", totalTokens);

        System.out.println("\nConcurrent processing allows Gollek to utilize all CPU cores for heavy NLP tasks.");
    }
}
