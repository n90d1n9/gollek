package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFParser;
import tech.kayys.gollek.gguf.loader.GGUFReader;
import tech.kayys.gollek.gguf.runtime.GgufBudget;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProbe;
import tech.kayys.gollek.gguf.runtime.GgufTensorOps;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;

import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java GGUF backend used for the actively working native-loader path.
 *
 * <p>It intentionally refuses generation for now. That keeps the production
 * runner from silently taking the slow/wrong path while still exposing the
 * Java loader readiness profile for diagnostics and future engine work.</p>
 */
final class JavaNativeGgufBackend implements GgufBackend {
    private final GGUFModel model;
    private final GgufRuntimeProfile profile;
    private final GgufRuntimeProbe.PreparedMatrixCacheDecision preparedCacheDecision;

    JavaNativeGgufBackend(Path modelPath) throws Exception {
        long startNanos = System.nanoTime();
        this.model = loadModel(modelPath);
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
    public String name() {
        return "java";
    }

    @Override
    public Map<String, Object> metadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("backend", name());
        metadata.put("javaStatus", profile.javaStatus());
        metadata.put("architecture", profile.architecture());
        metadata.put("ggufVersion", profile.ggufVersion());
        metadata.put("tensorCount", profile.tensorCount());
        metadata.put("decoderTensorRatio", profile.decoderTensorRatio());
        metadata.put("missingDecoderTensorCount", profile.missingDecoderTensorCount());
        metadata.put("malformedDecoderTensorCount", profile.malformedDecoderTensorCount());
        metadata.put("missingDecoderTensorExamples", profile.missingDecoderTensorExamples());
        metadata.put("malformedDecoderTensorExamples", profile.malformedDecoderTensorExamples());
        metadata.put("knownTensorTypeRatio", profile.knownTensorTypeRatio());
        metadata.put("preparedMatrixCachePlan", preparedCacheDecision.plan().compactSummary());
        metadata.put("preparedMatrixCache", preparedCacheDecision.compactSummary());
        metadata.put("preparedMatrixCacheMode", preparedCacheDecision.mode());
        metadata.put("preparedMatrixCacheEstimatedBytes", preparedCacheDecision.plan().estimatedPreparedBytes());
        metadata.put("preparedMatrixCacheBytes", preparedCacheDecision.stats().cacheBytes());
        metadata.put("preparedMatrixCacheEntries", preparedCacheDecision.stats().cacheEntries());
        metadata.put("loaderReady", profile.knownTensorTypeRatio() > 0.0d);
        metadata.put("decoderTensorsReady", profile.decoderTensorSetComplete());
        metadata.put("rowDotReady", profile.rowDotPrimitivesReady());
        metadata.put("generationReady", false);
        if (profile.modelConfig() != null) {
            metadata.put("modelType", profile.modelConfig().getModelType());
            metadata.put("layers", profile.modelConfig().getNumHiddenLayers());
            metadata.put("hiddenSize", profile.modelConfig().getHiddenSize());
            metadata.put("attentionHeads", profile.modelConfig().getNumAttentionHeads());
            metadata.put("kvHeads", profile.modelConfig().getResolvedNumKvHeads());
            metadata.put("headDim", profile.modelConfig().getResolvedHeadDim());
            metadata.put("contextLength", profile.modelConfig().getMaxPositionEmbeddings());
            metadata.put("vocabSize", profile.modelConfig().getVocabSize());
        }
        return Map.copyOf(metadata);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        return RunnerResult.failed(
                "Java-native GGUF generation is not enabled yet (" + profile.javaStatus()
                        + "; preparedMatrixCache=" + preparedCacheDecision.compactSummary()
                        + "). Use gguf.backend=llamacpp or the CLI fast runner for inference.");
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

    private static GGUFModel loadModel(Path modelPath) throws Exception {
        Arena arena = Arena.ofShared();
        try (GGUFReader reader = new GGUFReader(modelPath, arena)) {
            return new GGUFParser().parse(reader.segment(), arena);
        } catch (Exception exception) {
            arena.close();
            throw exception;
        } catch (Error error) {
            arena.close();
            throw error;
        }
    }

    private static long autoPrepareBudgetBytes() {
        return GgufBudget.byteSizeProperty(
                "gollek.gguf.java_native.auto_prepare_budget_bytes",
                GgufBudget.defaultAutoPrepareBytes());
    }

}
