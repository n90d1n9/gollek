///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-recursive-reasoning:0.1.0-SNAPSHOT

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.ml.reasoning.InMemoryRecursiveReasoningSession;
import tech.kayys.gollek.ml.reasoning.RecursiveReasoningConfig;
import tech.kayys.gollek.ml.reasoning.RecursiveReasoningContext;
import tech.kayys.gollek.ml.reasoning.RecursiveReasoningReport;
import tech.kayys.gollek.ml.reasoning.RecursiveReasoningState;
import tech.kayys.gollek.ml.reasoning.RecursiveReasoningTransitionResult;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Tiny GRAM-style recursive reasoning demo for the new module family.
 */
public class trainer_recursive_reasoning_demo {

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        String taskId = args.length > 0 ? args[0] : "sudoku-demo";
        Path outputDir = args.length > 1
                ? Path.of(args[1])
                : Path.of("trainer_checkpoints", "recursive_reasoning_demo");
        recreateDirectory(outputDir);

        RecursiveReasoningConfig config = RecursiveReasoningConfig.builder()
                .supervisionSteps(2)
                .transitionsPerSupervisionStep(2)
                .parallelSamples(3)
                .seed(2026L)
                .checkpointDir(outputDir)
                .build();

        Tensor latent = new DummyTensor();
        RecursiveReasoningContext context = new RecursiveReasoningContext(taskId, latent, config);
        InMemoryRecursiveReasoningSession session = new InMemoryRecursiveReasoningSession(
                context,
                latent,
                (previous, ignored) -> {
                    int nextTransition = previous.transitionIndex() + 1;
                    double reward = (previous.sampleIndex() * 0.5d) + (previous.supervisionStep() * 0.25d);
                    return new RecursiveReasoningTransitionResult(
                            new RecursiveReasoningState(
                                    "state-s" + previous.sampleIndex()
                                            + "-e" + previous.supervisionStep()
                                            + "-t" + nextTransition,
                                    previous.supervisionStep(),
                                    nextTransition,
                                    previous.sampleIndex(),
                                    previous.latentState(),
                                    Map.of(
                                            "taskId", taskId,
                                            "heuristic", "demo-reward")),
                            -0.25d * nextTransition,
                            reward);
                });

        TrainingSummary summary = session.fit();
        RecursiveReasoningReport report = session.report();
        Path reportFile = outputDir.resolve("recursive-reasoning-report.json");
        Files.writeString(reportFile, toJson(report));

        System.out.println("====================================================");
        System.out.println(" Gollek Recursive Reasoning Demo (JBang)");
        System.out.println("====================================================");
        System.out.println("taskId=" + taskId);
        System.out.println("outputDir=" + outputDir.toAbsolutePath().normalize());
        System.out.println("parallelSamples=" + config.parallelSamples());
        System.out.println("supervisionSteps=" + config.supervisionSteps());
        System.out.println("transitionsPerSupervisionStep=" + config.transitionsPerSupervisionStep());
        System.out.println("selectedTrajectoryIndex=" + report.selectedTrajectoryIndex());
        System.out.println("selectedStateId=" + report.selectedStateId());
        System.out.println("selectedRewardScore=" + report.selectedRewardScore());
        System.out.println("selectedCumulativeLogProbability=" + report.selectedCumulativeLogProbability());
        System.out.println("trajectoryCount=" + report.summary().exploredTrajectoryCount());
        System.out.println("trainingMetadata=" + summary.metadata());
        System.out.println("reportFile=" + reportFile.toAbsolutePath().normalize());
        System.out.println("reportBytes=" + Files.size(reportFile));
        System.out.println("reportPreview=");
        Files.readString(reportFile).lines().limit(12).forEach(line -> System.out.println("  " + line));
    }

    private static String toJson(RecursiveReasoningReport report) {
        return """
                {
                  "reportVersion": "%s",
                  "familyId": "%s",
                  "taskId": "%s",
                  "config": {
                    "transitionsPerSupervisionStep": %d,
                    "supervisionSteps": %d,
                    "parallelSamples": %d,
                    "seed": %d
                  },
                  "summary": {
                    "exploredTrajectoryCount": %d,
                    "completedTrajectoryCount": %d,
                    "selectedStateId": "%s"
                  },
                  "selectedTrajectoryIndex": %d,
                  "selectedCumulativeLogProbability": %.4f,
                  "selectedRewardScore": %.4f
                }
                """.formatted(
                report.reportVersion(),
                report.familyId(),
                report.taskId(),
                report.config().transitionsPerSupervisionStep(),
                report.config().supervisionSteps(),
                report.config().parallelSamples(),
                report.config().seed(),
                report.summary().exploredTrajectoryCount(),
                report.summary().completedTrajectoryCount(),
                report.selectedStateId(),
                report.selectedTrajectoryIndex(),
                report.selectedCumulativeLogProbability(),
                report.selectedRewardScore() == null ? Double.NaN : report.selectedRewardScore());
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

    private static final class DummyTensor implements Tensor {
        @Override
        public Shape shape() { return new Shape(1, 1); }
        @Override
        public DeviceType device() { return DeviceType.CPU; }
        @Override
        public DType dtype() { return DType.F32; }
        @Override
        public ComputeBackend backend() { return null; }
        @Override
        public Tensor add(Tensor other) { return this; }
        @Override
        public Tensor sub(Tensor other) { return this; }
        @Override
        public Tensor mul(Tensor other) { return this; }
        @Override
        public Tensor mul(float scalar) { return this; }
        @Override
        public Tensor div(float scalar) { return this; }
        @Override
        public Tensor matmul(Tensor other) { return this; }
        @Override
        public Tensor reshape(long... newShape) { return this; }
        @Override
        public Tensor softmax() { return this; }
        @Override
        public Tensor slice(long[] offsets, long[] sizes) { return this; }
        @Override
        public Tensor pow(float exponent) { return this; }
        @Override
        public Tensor mean() { return this; }
        @Override
        public Tensor abs() { return this; }
        @Override
        public Tensor crossEntropy(Tensor target) { return this; }
        @Override
        public Tensor binaryCrossEntropy(Tensor target) { return this; }
        @Override
        public Tensor div(Tensor other) { return this; }
        @Override
        public Tensor add(float scalar) { return this; }
        @Override
        public Tensor zerosLike() { return this; }
        @Override
        public Tensor sqrt() { return this; }
        @Override
        public Tensor cast(DType dtype) { return this; }
        @Override
        public Tensor to(DeviceType device) { return this; }
        @Override
        public float item() { return 0f; }
        @Override
        public void backward() {}
        @Override
        public Tensor grad() { return this; }
        @Override
        public void setGrad(Tensor grad) {}
        @Override
        public boolean requiresGrad() { return false; }
        @Override
        public void setRequiresGrad(boolean requiresGrad) {}
        @Override
        public Tensor relu() { return this; }
        @Override
        public Tensor sigmoid() { return this; }
        @Override
        public Tensor tanh() { return this; }
        @Override
        public Tensor log() { return this; }
        @Override
        public Tensor exp() { return this; }
        @Override
        public Tensor silu() { return this; }
        @Override
        public Tensor flatten() { return this; }
        @Override
        public Tensor unsqueeze(int dim) { return this; }
        @Override
        public Tensor squeeze() { return this; }
        @Override
        public Tensor transpose() { return this; }
        @Override
        public Tensor transpose(int dim0, int dim1) { return this; }
        @Override
        public long numel() { return 1L; }
    }
}
