import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class DumpVocab {
    public static void main(String[] args) throws Exception {
        String path = "/Users/bhangun/.gollek/models/gguf/google_gemma-4-E4B-it.gguf";
        GGUFModel model = GGUFLoader.load(Path.of(path));
        List<String> tokens = (List<String>) model.metadata().get("tokenizer.ggml.tokens");
        if (tokens == null) tokens = (List<String>) model.metadata().get("tokenizer.tokens");
        
        System.out.println("Vocab size: " + tokens.size());
        for (int i = 0; i < Math.min(1000, tokens.size()); i++) {
            String t = tokens.get(i);
            if (t.contains("turn") || t.contains("start") || t.contains("end")) {
                System.out.println("Token [" + i + "]: '" + t + "'");
            }
        }
    }
}
