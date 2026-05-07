package tech.kayys.gollek.core.tensor;

import java.util.Random;

/**
 * Factory for creating tensors.
 * This is used by static methods in the {@link Tensor} interface.
 */
public final class TensorFactory {
    private static final Random RNG = new Random();

    public static Tensor randn(long... shape) {
        // This should delegate to the active default backend.
        // For now, we just return null to allow compilation, 
        // but it should eventually use a backend registry.
        return null; 
    }

    public static Tensor zeros(long... shape) {
        return null;
    }

    public static Tensor ones(long... shape) {
        return null;
    }

    public static Tensor full(float value, long... shape) {
        return null;
    }

    public static Tensor of(float[] data, long... shape) {
        return null;
    }
}
