package tech.kayys.gollek.ml.nlp;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.ml.tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Utility class for vector operations using JDK 25 Vector API and FFM API.
 */
public class VectorUtils {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static float cosineSimilarity(Tensor a, Tensor b) {
        return cosineSimilarity(a.data(), b.data());
    }

    public static float cosineSimilarity(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vector dimensions must match");
        }

        float dotProduct = dotProduct(v1, v2);
        float normA = norm(v1);
        float normB = norm(v2);

        if (normA == 0 || normB == 0) return 0.0f;
        return dotProduct / (float)(Math.sqrt(normA) * Math.sqrt(normB));
    }

    public static float dotProduct(float[] a, float[] b) {
        float dot = 0.0f;
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);

        for (; i < upperBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            dot += va.mul(vb).reduceLanes(VectorOperators.ADD);
        }

        for (; i < a.length; i++) {
            dot += a[i] * b[i];
        }

        return dot;
    }

    public static float dotProduct(float[] a, MemorySegment b, long offsetBytes) {
        float dot = 0.0f;
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);

        for (; i < upperBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromMemorySegment(SPECIES, b, offsetBytes + (long) i * ValueLayout.JAVA_FLOAT.byteSize(), ByteOrder.nativeOrder());
            dot += va.mul(vb).reduceLanes(VectorOperators.ADD);
        }

        for (; i < a.length; i++) {
            dot += a[i] * b.get(ValueLayout.JAVA_FLOAT, offsetBytes + (long) i * ValueLayout.JAVA_FLOAT.byteSize());
        }

        return dot;
    }

    private static float norm(float[] a) {
        float sumSq = 0.0f;
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);

        for (; i < upperBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            sumSq += va.mul(va).reduceLanes(VectorOperators.ADD);
        }

        for (; i < a.length; i++) {
            sumSq += a[i] * a[i];
        }

        return sumSq;
    }
}
