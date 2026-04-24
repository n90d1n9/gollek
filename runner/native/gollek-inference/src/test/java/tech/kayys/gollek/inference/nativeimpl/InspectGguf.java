import java.nio.file.Path;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import tech.kayys.gollek.gguf.loader.GGUFModel;

public class InspectGguf {
    public static void main(String[] args) throws Exception {
        GGUFModel model = GGUFLoader.loadModel(Path.of(System.getProperty("user.home") + "/.gollek/models/gguf/google_gemma-4-E4B-it.gguf"));
        System.out.println("Metadata keys:");
        model.metadata().keySet().forEach(System.out::println);
        System.out.println("Architecture: " + model.metadata().get("general.architecture"));
        System.out.println("Block count: " + model.metadata().get("gemma.block_count"));
        System.out.println("Tensors (first 20):");
        model.tensors().stream().limit(20).forEach(t -> System.out.println(t.name()));
    }
}
