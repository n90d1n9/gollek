package tech.kayys.gollek.inference.nativeimpl;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import java.nio.file.Paths;

public class MetadataNames {
    public static void main(String[] args) throws Exception {
        String modelPath = System.getProperty("user.home") + "/.gollek/models/gguf/google_gemma-4-E4B-it.gguf";
        try (GGUFModel model = GGUFLoader.loadModel(Paths.get(modelPath))) {
            model.metadata().forEach((k, v) -> {
                if (k.contains("soft") || k.contains("cap") || k.contains("gemma2")) {
                    System.out.println(k + " = " + v);
                }
            });
        }
    }
}
