package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.tokenizer.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.ModelConfig;
import tech.kayys.gollek.tokenizer.spi.TokenizerType;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Factory for creating Language pipelines.
 */
public class LanguageFactory {

    public static Language load(String modelName) {
        String langCode = modelName.split("_")[0];
        Path tokenizerPath = Paths.get("models", modelName, "tokenizer.json");
        
        Language.Tokenizer tokenizer;
        try {
            var modelConfig = new ModelConfig(TokenizerType.BPE, tokenizerPath);
            var gollekTokenizer = TokenizerFactory.create(modelConfig);
            
            tokenizer = text -> {
                Doc doc = new Doc(text);
                long[] ids = gollekTokenizer.encode(text, EncodeOptions.defaultOptions());
                
                // Simplified mapping for demonstration
                String[] words = text.split("\\s+");
                for (int i = 0; i < words.length; i++) {
                    doc.addToken(Token.builder()
                            .doc(doc)
                            .i(i)
                            .text(words[i])
                            .isAlpha(words[i].matches("[a-zA-Z]+"))
                            .build());
                }
                return doc;
            };
        } catch (Exception e) {
            tokenizer = text -> {
                Doc doc = new Doc(text);
                String[] words = text.split("\\s+");
                for (int i = 0; i < words.length; i++) {
                    doc.addToken(Token.builder().doc(doc).i(i).text(words[i]).build());
                }
                return doc;
            };
        }

        Language nlp = new Language(langCode, tokenizer);

        try {
            EmbeddingPipeline ep = new EmbeddingPipeline(modelName);
            nlp.addPipe(new EmbeddingProcessor(ep));
        } catch (Exception e) {
            // No embedding model available
        }

        return nlp;
    }
}
