package tech.kayys.gollek.inference.nativeimpl;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import java.nio.file.Paths;

public class TensorNames {
    public static void main(String[] args) throws Exception {
        String modelPath = System.getProperty("user.home") + "/.gollek/models/gguf/google_gemma-4-E4B-it.gguf";
        try (GGUFModel model = GGUFLoader.loadModel(Paths.get(modelPath))) {
            model.tensors().forEach(t -> {
                if (t.name().contains("output") || t.name().contains("lm_head") || t.name().contains("token_embd")) {
                    System.out.println(t.name() + " -> " + java.util.Arrays.toString(t.shape()) + " (type=" + t.typeId() + ")");
                }
            });
        }
    }
}
