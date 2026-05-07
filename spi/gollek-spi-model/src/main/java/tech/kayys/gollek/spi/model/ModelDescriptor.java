package tech.kayys.gollek.spi.model;
import tech.kayys.gollek.spi.spec.*;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.model.ModelFormat;

import java.net.URI;
import java.util.Map;

public record ModelDescriptor(
        String id,
        String format, // gguf, onnx, safetensors, triton, etc
        URI source,
        Map<String, String> metadata) {
}
