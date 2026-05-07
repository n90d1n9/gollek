package tech.kayys.gollek.spi.model;
import tech.kayys.gollek.spi.spec.*;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.model.ModelFormat;

import java.nio.file.Path;
import java.util.Map;

public record ModelArtifact(
        Path path,
        String checksum,
        Map<String, String> metadata) {
}
