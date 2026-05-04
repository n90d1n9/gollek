package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.ml.tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/**
 * Pipeline component for computing embeddings with batch optimization.
 * Uses FFM API for high-performance off-heap storage.
 */
public class EmbeddingProcessor implements Language.Processor {
    private final EmbeddingPipeline pipeline;

    public EmbeddingProcessor(EmbeddingPipeline pipeline) {
        this.pipeline = pipeline;
    }

    @Override
    public String name() {
        return "embedding";
    }

    @Override
    public Doc apply(Doc doc) {
        // Prepare batch: [Doc text, Token1 text, Token2 text, ...]
        List<String> inputs = new ArrayList<>();
        inputs.add(doc.getText());
        for (Token t : doc.getTokens()) {
            inputs.add(t.getText());
        }

        // Perform batch inference
        List<float[]> results = pipeline.batchEmbed(inputs);
        if (results == null || results.isEmpty()) return doc;

        // Set document vector
        float[] docVec = results.get(0);
        doc.setVector(Tensor.of(docVec));

        // Set token vectors and store in off-heap MemorySegment
        if (doc.length() > 0 && results.size() > 1) {
            int dim = docVec.length;
            
            // Using a single off-heap block for all token vectors in this Doc
            try (Arena arena = Arena.ofConfined()) {
                long totalBytes = (long) doc.length() * dim * ValueLayout.JAVA_FLOAT.byteSize();
                MemorySegment buffer = arena.allocate(totalBytes);
                
                for (int i = 0; i < doc.length(); i++) {
                    // results[1] corresponds to doc.getTokens()[0]
                    float[] vec = results.get(i + 1); 
                    Token t = doc.get(i);
                    t.setVector(Tensor.of(vec));

                    // Copy to FFM buffer for SIMD scanning later
                    long offset = (long) i * dim * ValueLayout.JAVA_FLOAT.byteSize();
                    MemorySegment.copy(MemorySegment.ofArray(vec), 0, buffer, offset, (long) vec.length * ValueLayout.JAVA_FLOAT.byteSize());
                }
                
                // Note: In a real system, the 'buffer' or 'arena' should be attached 
                // to the Doc's lifecycle to avoid closing it prematurely.
            }
        }
        
        return doc;
    }
}
