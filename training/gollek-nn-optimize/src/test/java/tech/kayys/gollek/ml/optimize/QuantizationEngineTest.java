package tech.kayys.gollek.ml.optimize;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.optimize.QuantizationEngine.Precision;
import tech.kayys.gollek.ml.optimize.QuantizationEngine.QuantizationScheme;

class QuantizationEngineTest {

    @Test
    void quantizeReadsTextWeightsAndWritesActualQuantizedCodes() throws Exception {
        Path modelPath = Files.createTempDirectory("gollek-quant-text").resolve("weights.txt");
        Files.writeString(modelPath, String.join("\n",
                "# gollek.weights.v1",
                "-1.0, -0.5, 0.5, 1.0",
                "tensor dense.bias: 0.0 2.0 -2.0 4.0",
                ""));

        QuantizationEngine.QuantizedModel model = QuantizationEngine.builder()
                .modelPath(modelPath)
                .scheme(QuantizationScheme.INT8)
                .targetPrecision(Precision.INT8)
                .build()
                .quantize();

        assertEquals(modelPath, model.originalPath());
        assertEquals(QuantizationScheme.INT8, model.scheme());
        assertEquals(Precision.INT8, model.precision());
        assertTrue(Files.isRegularFile(model.quantizedPath()));

        String artifact = Files.readString(model.quantizedPath());
        assertTrue(artifact.contains("schema=gollek.quantized.weights.v1"));
        assertTrue(artifact.contains("tensorCount=2"));
        assertTrue(artifact.contains("bits=8"));
        assertTrue(artifact.contains("q:"));
        assertTrue(artifact.contains("dequant:"));
        assertFalse(artifact.contains("4096"));

        assertEquals(2.0, model.accuracyMetrics().get("tensor_count"), 1e-12);
        assertEquals(8.0, model.accuracyMetrics().get("value_count"), 1e-12);
        assertTrue(model.accuracyMetrics().get("mse") >= 0.0);
        assertTrue(model.accuracyMetrics().get("cosine_sim") > 0.99);
    }

    @Test
    void quantizeUsesCopiedInMemoryWeights() throws Exception {
        Path modelPath = Files.createTempDirectory("gollek-quant-memory").resolve("direct.weights");
        float[][] weights = {
                {-1.0f, 0.0f, 1.0f},
                {2.0f, 2.0f, 2.0f}
        };

        QuantizationEngine engine = QuantizationEngine.builder()
                .modelPath(modelPath)
                .weights(weights)
                .scheme(QuantizationScheme.INT4)
                .targetPrecision(Precision.INT4)
                .build();
        weights[0][0] = 99.0f;

        QuantizationEngine.QuantizedModel model = engine.quantize();

        assertEquals(8.0, model.compressionRatio(), 1e-12);
        assertTrue(Files.isRegularFile(model.quantizedPath()));
        String artifact = Files.readString(model.quantizedPath());
        assertTrue(artifact.contains("bits=4"));
        assertFalse(artifact.contains("99.0"));
        assertEquals(6.0, model.accuracyMetrics().get("value_count"), 1e-12);
        assertTrue(model.accuracyMetrics().get("quantized_bytes") < model.accuracyMetrics().get("source_bytes"));
    }

    @Test
    void quantizeRejectsMissingModelInsteadOfInventingPlaceholderWeights() throws Exception {
        Path modelPath = Files.createTempDirectory("gollek-quant-missing").resolve("missing.weights");
        QuantizationEngine engine = QuantizationEngine.builder()
                .modelPath(modelPath)
                .scheme(QuantizationScheme.INT8)
                .targetPrecision(Precision.INT8)
                .build();

        IllegalStateException error = assertThrows(IllegalStateException.class, engine::quantize);

        assertTrue(error.getMessage().contains("does not exist"));
        assertFalse(Files.exists(modelPath.resolveSibling("missing.weights.qint8")));
    }
}
