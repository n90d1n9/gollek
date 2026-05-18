package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;

import java.nio.file.Path;
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
    private final GgufRuntimeProfile profile;

    JavaNativeGgufBackend(Path modelPath) throws Exception {
        this.profile = GgufRuntimeProfile.load(modelPath);
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
        metadata.put("loaderReady", profile.knownTensorTypeRatio() > 0.0d);
        metadata.put("decoderTensorsReady", profile.decoderTensorSetComplete());
        metadata.put("rowDotReady", profile.rowDotPrimitivesReady());
        metadata.put("generationReady", false);
        if (profile.modelConfig() != null) {
            metadata.put("modelType", profile.modelConfig().modelType());
            metadata.put("layers", profile.modelConfig().numHiddenLayers());
            metadata.put("hiddenSize", profile.modelConfig().hiddenSize());
            metadata.put("attentionHeads", profile.modelConfig().numAttentionHeads());
            metadata.put("kvHeads", profile.modelConfig().resolvedNumKvHeads());
            metadata.put("headDim", profile.modelConfig().resolvedHeadDim());
            metadata.put("contextLength", profile.modelConfig().maxPositionEmbeddings());
            metadata.put("vocabSize", profile.modelConfig().vocabSize());
        }
        return Map.copyOf(metadata);
    }

    @Override
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        return RunnerResult.failed(
                "Java-native GGUF generation is not enabled yet (" + profile.javaStatus()
                        + "). Use gguf.backend=llamacpp or the CLI fast runner for inference.");
    }

    @Override
    public void close() {
        // GgufRuntimeProfile closes the mmap reader after profiling; no live model handle is retained.
    }
}
