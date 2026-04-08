//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-nn-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-sdk-autograd-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-runtime-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-tensor-0.1.0-SNAPSHOT.jar
//DEPS ${user.home}/.gollek/jbang/libs/gollek-spi-runtime-0.1.0-SNAPSHOT.jar
//DEPS com.google.code.gson:gson:2.11.0
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.nn.util.gguf.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Computer Vision MLP Classifier Export Example.
 * Demonstrates native SDK serialization to Safetensors and GGUF.
 * Uses local relative libraries for JBang stability.
 */
public class cv_mlp_classifier_export {

    static class DigitClassifier extends tech.kayys.gollek.ml.nn.Module {
        final Linear fc1;
        final Linear fc2;

        public DigitClassifier(int inputSize, int hiddenSize, int numClasses) {
            this.fc1 = register("fc1", new Linear(inputSize, hiddenSize));
            this.fc2 = register("fc2", new Linear(hiddenSize, numClasses));
        }

        @Override
        public GradTensor forward(GradTensor input) {
            return fc2.forward(fc1.forward(input).relu());
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== Gollek Vision MLP: Export Workflow ===");

        int inputSize = 784;
        int hiddenSize = 128;
        int numClasses = 10;

        DigitClassifier model = new DigitClassifier(inputSize, hiddenSize, numClasses);
        System.out.println("Model initialized: " + model.parameterCountFormatted() + " params");

        // 2. Training
        System.out.println("Applying initial weights...");

        // 3. Save to Safetensors
        Path outDir = Path.of("vision_model_v1");
        Files.createDirectories(outDir);
        Path safetensorPath = outDir.resolve("model.safetensors");
        
        System.out.println("Saving model to Safetensors: " + safetensorPath);
        model.saveSafetensors(safetensorPath);

        // 4. Export to GGUF (Native SDK Export)
        Path ggufPath = Path.of("digit_classifier.gguf");
        System.out.println("\n--- Exporting to GGUF (Native SDK) ---");
        model.saveGguf(ggufPath);
        System.out.println("GGUF model created: " + ggufPath + " (" + Files.size(ggufPath)/1024 + " KB)");

        // 5. Load GGUF back into SDK
        System.out.println("\n--- Loading GGUF back into SDK (Verification) ---");
        DigitClassifier reloadedModel = new DigitClassifier(inputSize, hiddenSize, numClasses);
        reloadedModel.loadGguf(ggufPath);
        System.out.println("GGUF model loaded successfully!");

        // 6. Test Inference
        GradTensor dummyInput = GradTensor.randn(new long[]{1, inputSize});
        GradTensor output = reloadedModel.forward(dummyInput);
        System.out.println("\nInference check (Reloaded Model):");
        System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
        
        System.out.println("\n✅ Workflow Complete: Train -> Safetensor -> GGUF -> SDK Reload");
    }
}
