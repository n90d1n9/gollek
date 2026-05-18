///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-byte-latent:0.1.0-SNAPSHOT
//DEPS com.fasterxml.jackson.core:jackson-databind:2.20.1

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.bytelatent.ByteLatentGenerationResult;
import tech.kayys.gollek.ml.bytelatent.ByteLatentModel;
import tech.kayys.gollek.ml.bytelatent.ByteLatentModelSpec;
import tech.kayys.gollek.ml.bytelatent.ByteLatentTrainer;
import tech.kayys.gollek.ml.bytelatent.ByteLatentTrainerConfig;
import tech.kayys.gollek.ml.bytelatent.ReferenceByteLatentModel;
import tech.kayys.gollek.ml.bytelatent.TextByteSequenceDataset;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * JBang demo that runs a tiny byte-latent training session and then performs inference.
 */
public class trainer_byte_latent_train_infer_demo {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private static final String TRAIN_INFER_REPORT_FILE_NAME = "train-infer-report.json";

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Path checkpointDir = args.length > 0
                ? Path.of(args[0])
                : Path.of("trainer_checkpoints", "byte_latent_train_infer_demo");
        String prompt = args.length > 1 ? args[1] : "hi";
        int maxNewTokens = args.length > 2 ? Integer.parseInt(args[2]) : 4;
        recreateDirectory(checkpointDir);

        ByteLatentModelSpec modelSpec = new ByteLatentModelSpec(256, 96, 4, 4, 32);
        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of(
                "bismillah from gollek",
                "fast byte latent transformer demo",
                "tiny train then infer workflow",
                "terminal friendly reports matter"));

        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Train+Infer Demo (JBang)");
        System.out.println("====================================================");
        System.out.println("checkpointDir=" + checkpointDir.toAbsolutePath().normalize());
        System.out.println("prompt=" + prompt);
        System.out.println("maxNewTokens=" + maxNewTokens);

        TrainingSummary summary = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(ByteLatentTrainerConfig.builder()
                        .modelSpec(modelSpec)
                        .batchSize(2)
                        .windowLength(12)
                        .padTokenId(0)
                        .epochs(2)
                        .shuffle(false)
                        .seed(2026L)
                        .checkpointDir(checkpointDir)
                        .build())
                .build()
                .fit();

        ByteLatentModel model = new ReferenceByteLatentModel(modelSpec);
        int[] promptTokenIds = toUnsignedTokenIds(prompt);
        int nextToken = model.predictNextToken(promptTokenIds);
        ByteLatentGenerationResult generated = model.generate(promptTokenIds, maxNewTokens);

        Map<String, Object> metadata = summary.metadata();
        Path historyFile = checkpointDir.resolve("byte-latent-history.csv");
        Path reportFile = checkpointDir.resolve("byte-latent-report.json");
        Path trainInferReportFile = checkpointDir.resolve(TRAIN_INFER_REPORT_FILE_NAME);
        writeTrainInferReport(
                trainInferReportFile,
                checkpointDir,
                dataset,
                prompt,
                maxNewTokens,
                summary,
                historyFile,
                reportFile,
                nextToken,
                generated);

        System.out.println("----------------------------------------------------");
        System.out.println("datasetSize=" + dataset.size());
        System.out.println("epochCount=" + summary.epochCount());
        System.out.println("bestValidationLoss=" + summary.bestValidationLoss());
        System.out.println("latestTrainLoss=" + summary.latestTrainLoss());
        System.out.println("lossMode=" + metadata.get("lossMode"));
        System.out.println("modelClass=" + metadata.get("modelClass"));
        System.out.println("historyRowCount=" + metadata.get("historyRowCount"));
        System.out.println("globalStep=" + metadata.get("globalStep"));
        System.out.println("historyFile=" + historyFile.toAbsolutePath().normalize());
        System.out.println("reportFile=" + reportFile.toAbsolutePath().normalize());
        System.out.println("trainInferReportFile=" + trainInferReportFile.toAbsolutePath().normalize());

        System.out.println("----------------------------------------------------");
        System.out.println("Inference summary:");
        System.out.println("promptTokenIds=" + Arrays.toString(generated.promptTokenIds()));
        System.out.println("nextToken=" + nextToken);
        System.out.println("generatedTokenIds=" + Arrays.toString(generated.generatedTokenIds()));
        System.out.println("generatedText=" + generated.generatedText());
        System.out.println("combinedText=" + generated.combinedText());
        System.out.println("trainInferReportBytes=" + Files.size(trainInferReportFile));

        System.out.println("----------------------------------------------------");
        System.out.println("Follow-up examples:");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " status");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " history:summary");
        System.out.println("jbang trainer/trainer_byte_latent_train_infer_inspector.java "
                + trainInferReportFile.toAbsolutePath().normalize()
                + " overview");
        System.out.println("jbang trainer/trainer_byte_latent_train_infer_inspector.java "
                + trainInferReportFile.toAbsolutePath().normalize()
                + " ci");
        System.out.println("jbang trainer/trainer_byte_latent_infer_demo.java "
                + "\"" + prompt + "\" " + maxNewTokens);
    }

    private static void writeTrainInferReport(
            Path reportFile,
            Path checkpointDir,
            TextByteSequenceDataset dataset,
            String prompt,
            int maxNewTokens,
            TrainingSummary summary,
            Path historyFile,
            Path checkpointReportFile,
            int nextToken,
            ByteLatentGenerationResult generated) throws IOException {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("familyId", "fast-byte-latent-transformer");
        root.put("checkpointDir", checkpointDir.toAbsolutePath().normalize().toString());
        root.put("prompt", prompt);
        root.put("maxNewTokens", maxNewTokens);

        Map<String, Object> training = new LinkedHashMap<>();
        training.put("datasetSize", dataset.size());
        training.put("epochCount", summary.epochCount());
        training.put("bestValidationLoss", summary.bestValidationLoss());
        training.put("bestValidationEpoch", summary.bestValidationEpoch());
        training.put("latestTrainLoss", summary.latestTrainLoss());
        training.put("durationMs", summary.durationMs());
        training.put("metadata", summary.metadata());
        training.put("historyFile", historyFile.toAbsolutePath().normalize().toString());
        training.put("checkpointReportFile", checkpointReportFile.toAbsolutePath().normalize().toString());
        root.put("training", training);

        Map<String, Object> inference = new LinkedHashMap<>();
        inference.put("nextToken", nextToken);
        inference.put("promptTokenIds", generated.promptTokenIds());
        inference.put("generatedTokenIds", generated.generatedTokenIds());
        inference.put("combinedTokenIds", generated.combinedTokenIds());
        inference.put("generatedText", generated.generatedText());
        inference.put("combinedText", generated.combinedText());
        inference.put("metadata", generated.metadata());
        root.put("inference", inference);

        JSON.writeValue(reportFile.toFile(), root);
    }

    private static int[] toUnsignedTokenIds(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] tokenIds = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            tokenIds[i] = Byte.toUnsignedInt(bytes[i]);
        }
        return tokenIds;
    }

    private static void recreateDirectory(Path path) throws IOException {
        if (Files.exists(path)) {
            try (var stream = Files.walk(path)) {
                for (Path entry : stream.sorted(Comparator.reverseOrder()).toList()) {
                    Files.deleteIfExists(entry);
                }
            }
        }
        Files.createDirectories(path);
    }
}
