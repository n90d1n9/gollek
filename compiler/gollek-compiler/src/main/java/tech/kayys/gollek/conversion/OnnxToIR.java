package tech.kayys.gollek.conversion;

import tech.kayys.gollek.ir.*;
import tech.kayys.gollek.core.tensor.ModelWeightLoader;
import tech.kayys.gollek.core.tensor.WeightAdapter;
import java.nio.file.Path;

public final class OnnxToIR implements ModelWeightLoader {
    @Override
    public boolean supports(Path path) {
        return path.toString().endsWith(".onnx");
    }

    @Override
    public WeightAdapter load(Path path) {
        throw new UnsupportedOperationException("OnnxToIR not implemented");
    }
}