package tech.kayys.gollek.core.graph;

import tech.kayys.gollek.core.data.GData;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.core.backend.ComputeBackend;
import java.util.HashMap;
import java.util.Map;

public final class ExecutionContext {
    private final Map<String, GData> data = new HashMap<>();
    private final ComputeBackend backend;
    private final boolean gradEnabled;

    public ExecutionContext(ComputeBackend backend, boolean gradEnabled) {
        this.backend = backend;
        this.gradEnabled = gradEnabled;
    }

    public void put(GData d) {
        data.put(d.id(), d);
    }

    @SuppressWarnings("unchecked")
    public <T extends GData> T get(String id) {
        return (T) data.get(id);
    }

    public ComputeBackend backend() {
        return backend;
    }

    public boolean gradEnabled() {
        return gradEnabled;
    }
}