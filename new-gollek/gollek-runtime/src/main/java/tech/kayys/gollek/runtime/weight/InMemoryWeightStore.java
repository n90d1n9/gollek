package tech.kayys.gollek.runtime.weight;

import tech.kayys.gollek.core.tensor.Tensor;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public final class InMemoryWeightStore implements WeightStore {
    private final Map<String, Tensor> map = new ConcurrentHashMap<>();

    public void put(String key, Tensor tensor) {
        Objects.requireNonNull(key, "Weight key cannot be null");
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        map.put(key, tensor);
    }

    public void putAll(Map<String, Tensor> weights) {
        map.putAll(weights);
    }

    @Override
    public Tensor get(String key) {
        Tensor t = map.get(key);
        if (t == null) {
            throw new MissingWeightException(
                    String.format("Weight '%s' not found. Available keys: %s",
                            key, map.keySet().stream().limit(10).toList()));
        }
        return t;
    }

    @Override
    public boolean contains(String key) {
        return map.containsKey(key);
    }

    public int size() {
        return map.size();
    }

    public void clear() {
        map.clear();
    }
}

// Add this new exception class
class MissingWeightException extends RuntimeException {
    public MissingWeightException(String message) {
        super(message);
    }
}