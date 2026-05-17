package tech.kayys.gollek.ml.hub;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.gollek.safetensor.loader.SafetensorLoadResult;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Standalone bridge from safetensors files into GradTensor weights.
 */
final class SafeTensorBridge {

    private SafeTensorBridge() {
    }

    static Map<String, GradTensor> load(Path file) throws IOException {
        Map<String, GradTensor> weights = new LinkedHashMap<>();
        SafetensorFFMLoader loader = createStandaloneLoader();

        try (SafetensorLoadResult result = loader.load(file)) {
            for (String name : result.tensorNames()) {
                SafetensorTensor tensor = result.tensor(name);
                weights.put(name, GradTensor.of(tensor.toFloatArray(), tensor.shape()));
            }
        } catch (Exception e) {
            throw new IOException("Failed to load SafeTensors file: " + file, e);
        }

        return weights;
    }

    private static SafetensorFFMLoader createStandaloneLoader() {
        return new SafetensorFFMLoader();
    }
}
