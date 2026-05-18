///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-byte-latent:0.1.0-SNAPSHOT

import java.util.Arrays;
import java.util.Locale;
import tech.kayys.gollek.ml.bytelatent.ByteLatentGenerationResult;
import tech.kayys.gollek.ml.bytelatent.ByteLatentModel;
import tech.kayys.gollek.ml.bytelatent.ByteLatentModelSpec;
import tech.kayys.gollek.ml.bytelatent.ReferenceByteLatentModel;

/**
 * Tiny JBang demo for byte-latent prompt continuation and next-token inference.
 */
public class trainer_byte_latent_infer_demo {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        String prompt = args.length > 0 ? args[0] : "hi";
        int maxNewTokens = args.length > 1 ? Integer.parseInt(args[1]) : 4;

        ByteLatentModel model = new ReferenceByteLatentModel(
                new ByteLatentModelSpec(256, 64, 2, 4, 16));

        byte[] promptBytes = prompt.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int[] normalizedPrompt = new int[promptBytes.length];
        for (int i = 0; i < promptBytes.length; i++) {
            normalizedPrompt[i] = Byte.toUnsignedInt(promptBytes[i]);
        }

        int nextToken = model.predictNextToken(normalizedPrompt);
        ByteLatentGenerationResult generated = model.generate(normalizedPrompt, maxNewTokens);

        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Inference Demo (JBang)");
        System.out.println("====================================================");
        System.out.println("prompt=" + prompt);
        System.out.println("promptTokenIds=" + Arrays.toString(generated.promptTokenIds()));
        System.out.println("nextToken=" + nextToken);
        System.out.println("generatedTokenIds=" + Arrays.toString(generated.generatedTokenIds()));
        System.out.println("generatedText=" + generated.generatedText());
        System.out.println("combinedText=" + generated.combinedText());
        System.out.println("metadata=" + generated.metadata());
    }
}
