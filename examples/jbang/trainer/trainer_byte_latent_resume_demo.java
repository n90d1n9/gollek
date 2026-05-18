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
 * End-to-end JBang demo for byte-latent checkpoint resume flows.
 */
public class trainer_byte_latent_resume_demo {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Path checkpointDir = args.length > 0
                ? Path.of(args[0])
                : Path.of("trainer_checkpoints", "byte_latent_resume_demo");
        recreateDirectory(checkpointDir);

        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Resume Demo (JBang)");
        System.out.println("====================================================");
        System.out.println("checkpointDir=" + checkpointDir.toAbsolutePath().normalize());

        TextByteSequenceDataset dataset = dataset();
        ByteLatentModelSpec modelSpec = new ByteLatentModelSpec(256, 96, 4, 4, 128);

        TrainingSummary initial = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(ByteLatentTrainerConfig.builder()
                        .modelSpec(modelSpec)
                        .batchSize(2)
                        .windowLength(12)
                        .padTokenId(0)
                        .epochs(1)
                        .shuffle(false)
                        .seed(2026L)
                        .checkpointDir(checkpointDir)
                        .build())
                .lossEvaluator((batch, cfg, epoch, batchIndex) -> 0.8d + (batchIndex * 0.05d))
                .build()
                .fit();

        long initialHistoryLines = Files.readString(checkpointDir.resolve("byte-latent-history.csv"))
                .lines()
                .count();

        TrainingSummary resumed = ByteLatentTrainer.builder()
                .dataset(dataset)
                .config(ByteLatentTrainerConfig.builder()
                        .modelSpec(modelSpec)
                        .batchSize(2)
                        .windowLength(12)
                        .padTokenId(0)
                        .epochs(3)
                        .shuffle(false)
                        .seed(2026L)
                        .checkpointDir(checkpointDir)
                        .resumeFromCheckpoint(true)
                        .build())
                .lossEvaluator((batch, cfg, epoch, batchIndex) -> {
                    double epochBase = 0.8d - (epoch * 0.15d);
                    return Math.max(0.05d, epochBase + (batchIndex * 0.03d));
                })
                .build()
                .fit();

        Path historyFile = checkpointDir.resolve("byte-latent-history.csv");
        Path reportFile = checkpointDir.resolve("byte-latent-report.json");
        Map<String, Object> metadata = resumed.metadata();

        System.out.println("----------------------------------------------------");
        System.out.println("initialEpochs=" + initial.epochCount());
        System.out.println("initialHistoryLines=" + initialHistoryLines);
        System.out.println("resumedEpochs=" + resumed.epochCount());
        System.out.println("resumeRequested=" + metadata.get("resumeRequested"));
        System.out.println("resumeLoaded=" + metadata.get("resumeLoaded"));
        System.out.println("historyLoaded=" + metadata.get("historyLoaded"));
        System.out.println("historyRowCount=" + metadata.get("historyRowCount"));
        System.out.println("globalStep=" + metadata.get("globalStep"));
        System.out.println("bestValidationEpoch=" + resumed.bestValidationEpoch());
        System.out.println("bestValidationLoss=" + resumed.bestValidationLoss());
        System.out.println("latestTrainLoss=" + resumed.latestTrainLoss());
        System.out.println("historyFile=" + historyFile.toAbsolutePath().normalize());
        System.out.println("reportFile=" + reportFile.toAbsolutePath().normalize());
        System.out.println("historyPreview=");
        Files.readString(historyFile).lines().limit(6).forEach(line -> System.out.println("  " + line));

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
                + " overview json");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " status:ci");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " summary:metadata:globalStep");
        System.out.println("jbang trainer/trainer_byte_latent_history_inspector.java "
                + checkpointDir.toAbsolutePath().normalize()
                + " history:summary");
    }

    private static TextByteSequenceDataset dataset() {
        return new TextByteSequenceDataset(List.of(
                "bismillah from gollek",
                "fast byte latent transformer demo",
                "resume checkpoints should append history",
                "terminal friendly reports matter"));
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
