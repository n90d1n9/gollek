package tech.kayys.gollek.runtime.weight;

import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.memory.CpuBuffer;
import java.nio.file.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class LazyWeightStore implements WeightStore {
    private final Path weightDirectory;
    private final Map<String, Tensor> loaded = new ConcurrentHashMap<>();

    public LazyWeightStore(Path weightDirectory) {
        this.weightDirectory = weightDirectory;
    }

    @Override
    public Tensor get(String key) {
        // Check cache first
        if (loaded.containsKey(key)) {
            return loaded.get(key);
        }

        // Load from disk
        Path weightFile = weightDirectory.resolve(key + ".bin");
        if (!Files.exists(weightFile)) {
            throw new MissingWeightException(
                    "Weight not found: " + key + " at " + weightFile);
        }

        try {
            Tensor tensor = loadFromFile(weightFile);
            loaded.put(key, tensor);
            return tensor;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load weight: " + key, e);
        }
    }

    private Tensor loadFromFile(Path path) throws Exception {
        // Implementation depends on your serialization format
        byte[] data = Files.readAllBytes(path);
        // Parse shape and data...
        return new DefaultTensor(
                new Shape(1), // placeholder
                DType.FLOAT32,
                Device.CPU,
                new CpuBuffer(data.length),
                null);
    }

    @Override
    public boolean contains(String key) {
        if (loaded.containsKey(key))
            return true;
        return Files.exists(weightDirectory.resolve(key + ".bin"));
    }
}