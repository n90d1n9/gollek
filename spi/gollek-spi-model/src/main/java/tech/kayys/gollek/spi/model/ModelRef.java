package tech.kayys.gollek.spi.model;
import tech.kayys.gollek.spi.spec.*;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.model.ModelFormat;

import java.util.Map;

public record ModelRef(
        String scheme, // hf, local, s3, git, http, custom
        String namespace, // org/user
        String name,
        String version,
        Map<String, String> parameters) {
}
