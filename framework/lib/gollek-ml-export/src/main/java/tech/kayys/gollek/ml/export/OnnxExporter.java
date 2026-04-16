package tech.kayys.gollek.ml.export;

import tech.kayys.gollek.ml.export.onnx.OnnxExporter;
import tech.kayys.gollek.ml.nn.NNModule;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Thin bridge kept for {@link ModelExporter} API compatibility.
 * All logic lives in {@code gollek-ml-export-onnx} →
 * {@link tech.kayys.gollek.ml.export.onnx.OnnxExporter}.
 *
 * @deprecated Use {@link OnnxExporter} directly.
 */
@Deprecated(forRemoval = true)
public class OnnxExporterBridge {

    private final OnnxExporter delegate;

    public OnnxExporterBridge(NNModule model, long[] inputShape, Map<String, Object> metadata) {
        this.delegate = OnnxExporter.fromModel(model);
    }

    public void export(Path outputPath) throws IOException {
        delegate.export(outputPath);
    }
}
