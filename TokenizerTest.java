import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import java.nio.file.Path;

public class TokenizerTest {
    public static void main(String[] args) throws Exception {
        Path modelDir = Path.of("/Users/bhangun/.gollek/models/blobs/google/gemma-4-E2B-it/google__gemma-4-E2B-it");
        Tokenizer tokenizer = TokenizerFactory.load(modelDir, null);
        String prompt = "<start_of_turn>user\nwhere is bandung<end_of_turn>\n<start_of_turn>model\n";
        System.out.println("Prompt: " + prompt);
        System.out.println("Tokens: " + java.util.Arrays.toString(tokenizer.encode(prompt)));
    }
}
