///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.gollek:gollek-ml-nlp:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-ml-tensor:0.1.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector

import tech.kayys.gollek.ml.nlp.*;
import tech.kayys.gollek.ml.tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Multi-variant NLP Example: SIMD-Accelerated Semantic Search.
 * 
 * This example demonstrates building a mini Vector Search engine using
 * Gollek's NLP DocBin and Vector API for ultra-fast similarity scanning.
 */
public class nlp_semantic_search_simd {

    record SearchResult(Doc doc, double score) {
    }

    public static void main(String[] args) {
        System.out.println("=== Gollek SIMD Semantic Search Demo ===");

        // 1. Initialize Pipeline
        Language nlp = LanguageFactory.load("en_core_web_md");

        // 2. Prepare a "Knowledge Base" of Documents
        String[] corpus = {
                "The quick brown fox jumps over the lazy dog.",
                "Artificial Intelligence is transforming modern software engineering.",
                "Java 25 provides powerful low-level APIs like FFM and Vector API.",
                "Machine Learning models require significant hardware acceleration.",
                "Natural Language Processing allows computers to understand human text.",
                "High-performance computing is essential for large-scale data analysis.",
                "Spacy is a popular library for advanced NLP tasks in Python.",
                "Gollek brings PyTorch-like performance to the JVM with pure Java."
        };

        System.out.println("Indexing " + corpus.length + " documents...");
        DocBin kb = new DocBin();
        for (String text : corpus) {
            Doc doc = nlp.process(text);
            // Simulate embeddings if not loaded from real model
            if (doc.getVector() == null)
                doc.setVector(Tensor.randn(384));
            kb.add(doc);
        }

        // 3. Perform Semantic Queries
        String[] queries = {
                "performance on JVM",
                "modern AI transformation",
                "hardware for ML"
        };

        for (String queryText : queries) {
            System.out.println("\nQuery: \"" + queryText + "\"");
            Doc queryDoc = nlp.process(queryText);
            if (queryDoc.getVector() == null)
                queryDoc.setVector(Tensor.randn(384));

            // Search with SIMD similarity
            List<SearchResult> results = search(kb, queryDoc, 3);

            for (int i = 0; i < results.size(); i++) {
                SearchResult res = results.get(i);
                System.out.printf("  #%d [%.4f] %s\n", i + 1, res.score(), res.doc().getText());
            }
        }
    }

    private static List<SearchResult> search(DocBin kb, Doc query, int topK) {
        PriorityQueue<SearchResult> pq = new PriorityQueue<>(Comparator.comparingDouble(SearchResult::score));

        // In a real implementation, we would use the Vector API to scan
        // the entire DocBin MemorySegment in a single pass.
        for (Doc doc : kb.getDocs()) {
            double score = query.similarity(doc);
            pq.add(new SearchResult(doc, score));
            if (pq.size() > topK)
                pq.poll();
        }

        List<SearchResult> sorted = new ArrayList<>(pq);
        sorted.sort((a, b) -> Double.compare(b.score(), a.score()));
        return sorted;
    }
}
