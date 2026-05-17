package tech.kayys.gollek.export;

import tech.kayys.gollek.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * @deprecated Use {@link tech.kayys.gollek.ml.export.ModelExporter} or
 *             {@link tech.kayys.gollek.ml.Gollek.Export#model(NNModule)} instead.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class ModelExporter {

    private final tech.kayys.gollek.ml.export.ModelExporter delegate;

    private ModelExporter(Builder builder) {
        this.delegate = builder.delegate.build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public void toONNX(Path outputPath) throws IOException {
        delegate.toONNX(outputPath);
    }

    public void toONNX(String outputPath) throws IOException {
        delegate.toONNX(outputPath);
    }

    public void toGGUF(Path outputPath, Quantization quantization) throws IOException {
        delegate.toGGUF(outputPath, toCanonical(quantization));
    }

    public void toGGUF(String outputPath, Quantization quantization) throws IOException {
        delegate.toGGUF(outputPath, toCanonical(quantization));
    }

    public void toLiteRT(Path outputPath) throws IOException {
        delegate.toLiteRT(outputPath);
    }

    public void toLiteRT(String outputPath) throws IOException {
        delegate.toLiteRT(outputPath);
    }

    public long estimateSize() {
        return delegate.estimateSize();
    }

    private static tech.kayys.gollek.ml.export.ModelExporter.Quantization toCanonical(Quantization quantization) {
        return switch (quantization) {
            case FP16 -> tech.kayys.gollek.ml.export.ModelExporter.Quantization.FP16;
            case INT8 -> tech.kayys.gollek.ml.export.ModelExporter.Quantization.INT8;
            case INT4 -> tech.kayys.gollek.ml.export.ModelExporter.Quantization.INT4;
            case NF4 -> tech.kayys.gollek.ml.export.ModelExporter.Quantization.NF4;
            default -> tech.kayys.gollek.ml.export.ModelExporter.Quantization.NONE;
        };
    }

    public enum Quantization {
        NONE,
        FP16,
        INT8,
        INT4,
        NF4
    }

    public static class Builder {
        private final tech.kayys.gollek.ml.export.ModelExporter.Builder delegate =
                tech.kayys.gollek.ml.export.ModelExporter.builder();

        public Builder model(NNModule model) {
            delegate.model(model);
            return this;
        }

        public Builder inputShape(long... shape) {
            delegate.inputShape(shape);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            delegate.metadata(metadata);
            return this;
        }

        public ModelExporter build() {
            return new ModelExporter(this);
        }
    }
}
