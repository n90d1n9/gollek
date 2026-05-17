package tech.kayys.gollek.ml.export;

import tech.kayys.gollek.gguf.core.GgufExporter;
import tech.kayys.gollek.ml.export.litert.LiteRTExporter;
import tech.kayys.gollek.ml.export.onnx.OnnxExporter;
import tech.kayys.gollek.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Unified model exporter for deployment formats.
 */
public class ModelExporter {

    private final NNModule model;
    private final long[] inputShape;
    private final Map<String, Object> metadata;

    private ModelExporter(Builder builder) {
        this.model = builder.model;
        this.inputShape = builder.inputShape;
        this.metadata = builder.metadata;
    }

    public static Builder builder() {
        return new Builder();
    }

    public void toONNX(Path outputPath) throws IOException {
        OnnxExporter.fromModel(model, inputShape).export(outputPath);
    }

    public void toONNX(String outputPath) throws IOException {
        toONNX(Path.of(outputPath));
    }

    public void toGGUF(Path outputPath, Quantization quantization) throws IOException {
        GgufExporter.fromModel(model, metadata)
                .quantization(toGgufQuant(quantization))
                .export(outputPath);
    }

    public void toGGUF(String outputPath, Quantization quantization) throws IOException {
        toGGUF(Path.of(outputPath), quantization);
    }

    public void toLiteRT(Path outputPath) throws IOException {
        LiteRTExporter.fromModel(model, inputShape).export(outputPath);
    }

    public void toLiteRT(String outputPath) throws IOException {
        toLiteRT(Path.of(outputPath));
    }

    private static GgufExporter.Quantization toGgufQuant(Quantization quantization) {
        return switch (quantization) {
            case FP16 -> GgufExporter.Quantization.FP16;
            case INT8 -> GgufExporter.Quantization.INT8;
            case INT4 -> GgufExporter.Quantization.INT4;
            case NF4 -> GgufExporter.Quantization.NF4;
            default -> GgufExporter.Quantization.NONE;
        };
    }

    public long estimateSize() {
        return model.parameterCount() * 4L;
    }

    public enum Quantization {
        NONE,
        FP16,
        INT8,
        INT4,
        NF4
    }

    public static class Builder {
        private NNModule model;
        private long[] inputShape;
        private Map<String, Object> metadata = Map.of();

        public Builder model(NNModule value) {
            this.model = value;
            return this;
        }

        public Builder inputShape(long... value) {
            this.inputShape = value.clone();
            return this;
        }

        public Builder metadata(Map<String, Object> value) {
            this.metadata = value;
            return this;
        }

        public ModelExporter build() {
            return new ModelExporter(this);
        }
    }
}
