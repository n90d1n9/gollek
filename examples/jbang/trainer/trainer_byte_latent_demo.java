///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-byte-latent:0.1.0-SNAPSHOT

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.bytelatent.ByteLatentModelSpec;
import tech.kayys.gollek.ml.bytelatent.ByteLatentTrainer;
import tech.kayys.gollek.ml.bytelatent.ByteLatentTrainerConfig;
import tech.kayys.gollek.ml.bytelatent.TextByteSequenceDataset;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * End-to-end JBang demo for byte-latent checkpoint generation.
 */
public class trainer_byte_latent_demo {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Path checkpointDir = args.length > 0
                ? Path.of(args[0])
                : Path.of("trainer_checkpoints", "byte_latent_demo");
        recreateDirectory(checkpointDir);

        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Demo (JBang)");
        System.out.println("====================================================");
        System.out.println("checkpointDir=" + checkpointDir.toAbsolutePath().normalize());

        TextByteSequenceDataset dataset = new TextByteSequenceDataset(List.of(
                "bismillah from gollek",
                "fast byte latent transformer demo",
                "byte checkpoints should be easy to inspect",
                "terminal friendly reports matter"));

        ByteLatentTrainerConfig config = ByteLatentTrainerConfig.builder()
                .modelSpec(new ByteLatentModelSpec(256, 96, 4, 4, 128))
                .batchSize(2)
                .windowLength(12)
                .padTokenId(0)
                .epochs(3)
                .shuffle(false)
                .seed(2026L)
                .checkpointDir(checkpointDir)
                .build();

        TrainingSummary summary = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(config)
                .lossEvaluator((batch, trainerConfig, epoch, batchIndex) -> {
                    double epochBase = 0.9d - (epoch * 0.15d);
                    return Math.max(0.05d, epochBase + (batchIndex * 0.02d));
                })
                .build()
                .fit();

        Map<String, Object> metadata = summary.metadata();
        Path historyFile = checkpointDir.resolve("byte-latent-history.csv");
        Path reportFile = checkpointDir.resolve("byte-latent-report.json");

        System.out.println("----------------------------------------------------");
        System.out.println("datasetSize=" + dataset.size());
        System.out.println("epochCount=" + summary.epochCount());
        System.out.println("bestValidationLoss=" + summary.bestValidationLoss());
        System.out.println("bestValidationEpoch=" + summary.bestValidationEpoch());
        System.out.println("latestTrainLoss=" + summary.latestTrainLoss());
        System.out.println("historyRowCount=" + metadata.get("historyRowCount"));
        System.out.println("globalStep=" + metadata.get("globalStep"));
        System.out.println("historyFile=" + historyFile.toAbsolutePath().normalize());
        System.out.println("reportFile=" + reportFile.toAbsolutePath().normalize());
        System.out.println("historyBytes=" + Files.size(historyFile));
        System.out.println("reportBytes=" + Files.size(reportFile));
        System.out.println("historyPreview=");
        Files.readString(historyFile).lines().limit(5).forEach(line -> System.out.println("  " + line));

        System.out.println("----------------------------------------------------");
        System.out.println("Inspector examples:");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " status");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " health");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " ci");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " summary json");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " status:ci");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " history:summary");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " history:sort=-trainLoss:top=3 json");
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
