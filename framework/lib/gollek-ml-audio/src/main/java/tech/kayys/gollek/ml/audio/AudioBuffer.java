package tech.kayys.gollek.ml.audio;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * High-performance audio data container using FFM API (MemorySegment)
 * and Vector API for SIMD-accelerated preprocessing.
 *
 * <p>Supports zero-copy views and optimized signal manipulation.
 */
public class AudioBuffer {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private final MemorySegment segment;
    private final int sampleRate;
    private final long numSamples;

    public AudioBuffer(float[] samples, int sampleRate) {
        this.sampleRate = sampleRate;
        this.numSamples = samples.length;
        // Allocate off-heap using a shared arena for simplicity; 
        // in production, closer lifecycle management may be needed.
        this.segment = Arena.ofAuto().allocate(numSamples * ValueLayout.JAVA_FLOAT.byteSize());
        this.segment.copyFrom(MemorySegment.ofArray(samples));
    }

    public AudioBuffer(MemorySegment segment, int sampleRate, long numSamples) {
        this.segment = segment;
        this.sampleRate = sampleRate;
        this.numSamples = numSamples;
    }

    public long size() {
        return numSamples;
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public MemorySegment getSegment() {
        return segment;
    }

    /**
     * Normalize audio to [-1.0, 1.0] range using SIMD.
     */
    public void normalize() {
        float maxAbs = maxAbs();
        if (maxAbs > 0) {
            scale(1.0f / maxAbs);
        }
    }

    /**
     * Scales all samples by a factor using SIMD.
     */
    public void scale(float factor) {
        long byteSize = segment.byteSize();
        long floatCount = byteSize / ValueLayout.JAVA_FLOAT.byteSize();
        
        long i = 0;
        long bound = SPECIES.loopBound(floatCount);
        FloatVector vFactor = FloatVector.broadcast(SPECIES, factor);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(SPECIES, segment, i * ValueLayout.JAVA_FLOAT.byteSize(), java.nio.ByteOrder.nativeOrder());
            v.mul(vFactor).intoMemorySegment(segment, i * ValueLayout.JAVA_FLOAT.byteSize(), java.nio.ByteOrder.nativeOrder());
        }

        // Tail
        for (; i < floatCount; i++) {
            float val = segment.get(ValueLayout.JAVA_FLOAT, i * ValueLayout.JAVA_FLOAT.byteSize());
            segment.set(ValueLayout.JAVA_FLOAT, i * ValueLayout.JAVA_FLOAT.byteSize(), val * factor);
        }
    }

    /**
     * Finds the maximum absolute value in the buffer using SIMD.
     */
    public float maxAbs() {
        long byteSize = segment.byteSize();
        long floatCount = byteSize / ValueLayout.JAVA_FLOAT.byteSize();
        
        long i = 0;
        long bound = SPECIES.loopBound(floatCount);
        FloatVector vMax = FloatVector.broadcast(SPECIES, 0.0f);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(SPECIES, segment, i * ValueLayout.JAVA_FLOAT.byteSize(), java.nio.ByteOrder.nativeOrder());
            vMax = vMax.max(v.abs());
        }

        float max = vMax.reduceLanes(VectorOperators.MAX);
        
        // Tail
        for (; i < floatCount; i++) {
            float val = Math.abs(segment.get(ValueLayout.JAVA_FLOAT, i * ValueLayout.JAVA_FLOAT.byteSize()));
            if (val > max) max = val;
        }
        
        return max;
    }

    /**
     * Utility to convert to standard float array (copies data).
     */
    public float[] toArray() {
        return segment.toArray(ValueLayout.JAVA_FLOAT);
    }
    
    /**
     * Get a slice of the audio buffer without copying.
     */
    public AudioBuffer slice(long offset, long length) {
        if (offset < 0 || length < 0 || (offset + length) > numSamples) {
            throw new IndexOutOfBoundsException();
        }
        MemorySegment sliced = segment.asSlice(offset * ValueLayout.JAVA_FLOAT.byteSize(), length * ValueLayout.JAVA_FLOAT.byteSize());
        return new AudioBuffer(sliced, sampleRate, length);
    }
}
