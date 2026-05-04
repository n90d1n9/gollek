package tech.kayys.gollek.ml.runner;

import java.lang.foreign.MemorySegment;
import java.util.Map;

/**
 * Input to a model inference call.
 *
 * <p>Supports multiple input types:
 * <ul>
 *   <li>Raw float arrays (for simple tensors)</li>
 *   <li>MemorySegments (for zero-copy from native memory)</li>
 *   <li>Named tensors (for multi-input models)</li>
 * </ul>
 *
 * @since 0.3.0
 */
public final class InferenceInput {

    private final String name;
    private final long[] shape;
    private final InputDataType dtype;
    private final float[] floatData;
    private final int[] intData;
    private final MemorySegment nativeData;
    private final Map<String, Object> metadata;

    private InferenceInput(Builder builder) {
        this.name = builder.name;
        this.shape = builder.shape.clone();
        this.dtype = builder.dtype;
        this.floatData = builder.floatData != null ? builder.floatData.clone() : null;
        this.intData = builder.intData != null ? builder.intData.clone() : null;
        this.nativeData = builder.nativeData;
        this.metadata = builder.metadata != null ? Map.copyOf(builder.metadata) : Map.of();
    }

    /**
     * Creates an input from a float array.
     */
    public static InferenceInput fromFloats(float[] data, long... shape) {
        return builder()
            .data(data)
            .shape(shape)
            .dtype(InputDataType.FLOAT32)
            .build();
    }

    /**
     * Creates an input from an int array.
     */
    public static InferenceInput fromInts(int[] data, long... shape) {
        return builder()
            .data(data)
            .shape(shape)
            .dtype(InputDataType.INT32)
            .build();
    }

    /**
     * Creates an input from native memory.
     */
    public static InferenceInput fromNative(MemorySegment data, long[] shape, InputDataType dtype) {
        return builder()
            .nativeData(data)
            .shape(shape)
            .dtype(dtype)
            .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    // Getters
    public String name() { return name; }
    public long[] shape() { return shape.clone(); }
    public InputDataType dtype() { return dtype; }
    public float[] floatData() { return floatData != null ? floatData.clone() : null; }
    public int[] intData() { return intData != null ? intData.clone() : null; }
    public MemorySegment nativeData() { return nativeData; }
    public Map<String, Object> metadata() { return metadata; }

    public long numel() {
        long n = 1;
        for (long d : shape) n *= d;
        return n;
    }

    /**
     * Supported input data types.
     */
    public enum InputDataType {
        FLOAT32, FLOAT16, BFLOAT16,
        INT8, INT16, INT32, INT64,
        UINT8,
        BOOL
    }

    /**
     * Builder for InferenceInput.
     */
    public static class Builder {
        private String name = "input";
        private long[] shape = new long[0];
        private InputDataType dtype = InputDataType.FLOAT32;
        private float[] floatData;
        private int[] intData;
        private MemorySegment nativeData;
        private Map<String, Object> metadata;

        Builder() {}

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder shape(long... shape) {
            this.shape = shape;
            return this;
        }

        public Builder dtype(InputDataType dtype) {
            this.dtype = dtype;
            return this;
        }

        public Builder data(float[] data) {
            this.floatData = data;
            if (this.shape.length == 0) {
                this.shape = new long[]{data.length};
            }
            return this;
        }

        public Builder data(int[] data) {
            this.intData = data;
            if (this.shape.length == 0) {
                this.shape = new long[]{data.length};
            }
            return this;
        }

        public Builder nativeData(MemorySegment nativeData) {
            this.nativeData = nativeData;
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = metadata;
            return this;
        }

        public InferenceInput build() {
            return new InferenceInput(this);
        }
    }
}
