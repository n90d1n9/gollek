package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import tech.kayys.gollek.gguf.runtime.GgufBudget;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProbe;
import tech.kayys.gollek.gguf.runtime.GgufTensorOps;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Java-native GGUF backend scaffold backed by the active GGUF tensor primitives.
 */
public class JavaNativeGgufBackend implements GgufBackend {
    private final GGUFModel model;
    private final GgufRuntimeProfile profile;
    private final GgufRuntimeProbe.PreparedMatrixCacheDecision preparedCacheDecision;

    public JavaNativeGgufBackend(Path modelPath) throws Exception {
        long startNanos = System.nanoTime();
        this.model = GGUFLoader.loadModel(modelPath);
        long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
        this.profile = GgufRuntimeProfile.fromModel(model, Files.size(modelPath), loadMillis);
        this.preparedCacheDecision = prepareDecoderMatrixCaches(model);
    }

    JavaNativeGgufBackend(GGUFModel model, long modelBytes, long loadMillis) {
        this.model = model;
        this.profile = GgufRuntimeProfile.fromModel(model, modelBytes, loadMillis);
        this.preparedCacheDecision = prepareDecoderMatrixCaches(model);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        return RunnerResult.failed("Java-native GGUF generation is not enabled yet; "
                + profile.javaStatus()
                + "; preparedMatrixCache=" + preparedCacheDecision.compactSummary()
                + ". Use gguf.backend=llamacpp for generation until scheduler integration lands.");
    }

    @Override
    public void close() {
        GgufTensorOps.clearPreparedMatrixCaches(model);
        model.close();
    }

    GgufTensorOps.PreparedMatrixCachePlan preparedCachePlan() {
        return preparedCacheDecision.plan();
    }

    GgufTensorOps.PreparedMatrixCacheStats preparedCacheStats() {
        return preparedCacheDecision.stats();
    }

    String preparedCacheSummary() {
        return preparedCacheDecision.compactSummary();
    }

    private static GgufRuntimeProbe.PreparedMatrixCacheDecision prepareDecoderMatrixCaches(GGUFModel model) {
        int explicitMinRows = Math.max(0, Integer.getInteger("gollek.gguf.java_native.prepare_min_rows", 0));
        if (explicitMinRows > 0) {
            return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                    model,
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(
                            model,
                            explicitMinRows,
                            false,
                            1,
                            0L));
        }

        int autoMinRows = Math.max(1, Integer.getInteger("gollek.gguf.java_native.auto_prepare_min_rows", 32));
        long budgetBytes = autoPrepareBudgetBytes();
        return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                model,
                GgufRuntimeProbe.selectDecoderPreparedMatrixCache(
                        model,
                        0,
                        Boolean.parseBoolean(System.getProperty("gollek.gguf.java_native.auto_prepare", "true")),
                        autoMinRows,
                        budgetBytes));
    }

    private static long autoPrepareBudgetBytes() {
        return GgufBudget.byteSizeProperty(
                "gollek.gguf.java_native.auto_prepare_budget_bytes",
                GgufBudget.defaultAutoPrepareBytes());
    }

}
