package tech.kayys.gollek.core.tensor;

import java.nio.file.Path;

public interface ModelWeightLoader {
    boolean supports(Path path);

    WeightAdapter load(Path path);
}
