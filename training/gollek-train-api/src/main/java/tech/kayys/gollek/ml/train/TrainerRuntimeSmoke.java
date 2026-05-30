package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.train.data.DataLoader.Batch;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * End-to-end trainer smoke contract for tiny real-fit diagnostics.
 */
public final class TrainerRuntimeSmoke {
    private TrainerRuntimeSmoke() {
    }

    public static Result run() throws IOException {
        return run(Options.defaults());
    }

    public static Result run(Path checkpointDir) throws IOException {
        return run(Options.builder().checkpointDir(checkpointDir).build());
    }

    public static Result run(Options options) throws IOException {
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path checkpointDir = resolvedCheckpointDir(resolvedOptions.checkpointDir());
        List<Batch> train = regressionTrainBatches();
        List<Batch> validation = regressionValidationBatches();

        TrainingSummary firstRun = Gollek.DL.fit(
                new Linear(1, 1),
                train,
                validation,
                resolvedOptions.firstRunEpochs(),
                resolvedOptions.learningRate(),
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .checkpointDir(checkpointDir)
                        .device(resolvedOptions.device())
                        .failOnAcceleratorFallback(resolvedOptions.failOnAcceleratorFallback())
                        .build());

        TrainingSummary resumedRun = Gollek.DL.fit(
                new Linear(1, 1),
                train,
                validation,
                resolvedOptions.resumedEpochs(),
                resolvedOptions.learningRate(),
                Gollek.DL.TrainingPreset.REGRESSION_MSE_SGD,
                Gollek.DL.trainingOptions()
                        .checkpointDir(checkpointDir)
                        .resumeFromCheckpoint()
                        .device(resolvedOptions.device())
                        .failOnAcceleratorFallback(resolvedOptions.failOnAcceleratorFallback())
                        .build());

        return new Result(
                checkpointDir,
                firstRun,
                resumedRun,
                evaluate(checkpointDir, resolvedOptions, firstRun, resumedRun));
    }

    public static String renderJUnitXml(Result result) {
        Result resolvedResult = Objects.requireNonNull(result, "result must not be null");
        long failures = resolvedResult.checks().stream().filter(check -> !check.passed()).count();
        double seconds = Math.max(
                0.0,
                (resolvedResult.firstRun().durationMs() + resolvedResult.resumedRun().durationMs()) / 1000.0);
        StringBuilder xml = new StringBuilder();
        xml.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.append("<testsuite name=\"gollek.trainer.runtime.smoke\" tests=\"")
                .append(resolvedResult.checks().size())
                .append("\" failures=\"")
                .append(failures)
                .append("\" errors=\"0\" skipped=\"0\" time=\"")
                .append(String.format(java.util.Locale.ROOT, "%.3f", seconds))
                .append("\">\n");
        xml.append("  <properties>\n");
        xml.append("    <property name=\"checkpointDir\" value=\"")
                .append(xml(resolvedResult.checkpointDir().toString()))
                .append("\"/>\n");
        xml.append("    <property name=\"firstRunEpochs\" value=\"")
                .append(resolvedResult.firstRun().epochCount())
                .append("\"/>\n");
        xml.append("    <property name=\"resumedRunEpochs\" value=\"")
                .append(resolvedResult.resumedRun().epochCount())
                .append("\"/>\n");
        xml.append("  </properties>\n");
        for (Check check : resolvedResult.checks()) {
            xml.append("  <testcase classname=\"gollek.trainer.runtime.smoke\" name=\"")
                    .append(xml(check.name()))
                    .append("\" time=\"0.000\"");
            if (check.passed()) {
                xml.append("/>\n");
            } else {
                xml.append(">\n");
                xml.append("    <failure message=\"")
                        .append(xml(check.detail()))
                        .append("\">")
                        .append(xml(check.detail()))
                        .append("</failure>\n");
                xml.append("  </testcase>\n");
            }
        }
        xml.append("</testsuite>\n");
        return xml.toString();
    }

    public static String renderMarkdown(Result result) {
        Result resolvedResult = Objects.requireNonNull(result, "result must not be null");
        Map<String, Object> firstMetadata = resolvedResult.firstRun().metadata();
        Map<String, Object> resumedMetadata = resolvedResult.resumedRun().metadata();
        StringBuilder markdown = new StringBuilder();
        markdown.append("# Gollek Trainer Runtime Smoke\n\n");
        markdown.append("- Status: ").append(status(resolvedResult.passed())).append('\n');
        markdown.append("- Checkpoint dir: `").append(resolvedResult.checkpointDir()).append("`\n");
        markdown.append("- First run epochs: ").append(resolvedResult.firstRun().epochCount()).append('\n');
        markdown.append("- Resumed run epochs: ").append(resolvedResult.resumedRun().epochCount()).append('\n');
        markdown.append("- Requested device: ").append(markdownCell(value(resumedMetadata, "requestedDevice"))).append('\n');
        markdown.append("- Execution backend: ").append(markdownCell(value(resumedMetadata, "executionBackend"))).append('\n');
        markdown.append("- Execution accelerated: ").append(markdownCell(value(resumedMetadata, "executionAccelerated"))).append('\n');
        markdown.append("- Execution fallback: ").append(markdownCell(value(resumedMetadata, "executionFallback"))).append("\n\n");

        markdown.append("## Checks\n\n");
        markdown.append("| Check | Status | Detail |\n");
        markdown.append("| --- | --- | --- |\n");
        for (Check check : resolvedResult.checks()) {
            markdown.append("| ")
                    .append(markdownCell(check.name()))
                    .append(" | ")
                    .append(status(check.passed()))
                    .append(" | ")
                    .append(markdownCell(check.detail().isBlank() ? "-" : check.detail()))
                    .append(" |\n");
        }
        markdown.append('\n');

        if (!resolvedResult.passed()) {
            markdown.append("## Failures\n\n");
            for (Check failure : resolvedResult.failures()) {
                markdown.append("- `")
                        .append(markdownCell(failure.name()))
                        .append("`: ")
                        .append(markdownCell(failure.detail()))
                        .append('\n');
            }
            markdown.append('\n');
        }

        markdown.append("## Runtime Metadata\n\n");
        markdown.append("| Field | First run | Resumed run |\n");
        markdown.append("| --- | ---: | ---: |\n");
        appendMetadataRow(markdown, "Optimizer steps", firstMetadata, resumedMetadata, "optimizerStepCount");
        appendMetadataRow(markdown, "Requested device", firstMetadata, resumedMetadata, "requestedDevice");
        appendMetadataRow(markdown, "Execution backend", firstMetadata, resumedMetadata, "executionBackend");
        appendMetadataRow(markdown, "Execution accelerated", firstMetadata, resumedMetadata, "executionAccelerated");
        appendMetadataRow(markdown, "Execution fallback", firstMetadata, resumedMetadata, "executionFallback");
        return markdown.toString();
    }

    public record Options(
            Path checkpointDir,
            String device,
            boolean failOnAcceleratorFallback,
            int firstRunEpochs,
            int resumedEpochs,
            float learningRate) {
        public Options {
            checkpointDir = checkpointDir == null ? null : checkpointDir.toAbsolutePath().normalize();
            device = TrainerAccelerationPolicy.normalizeDevice(device);
            if (firstRunEpochs <= 0) {
                throw new IllegalArgumentException("firstRunEpochs must be positive");
            }
            if (resumedEpochs <= firstRunEpochs) {
                throw new IllegalArgumentException("resumedEpochs must be greater than firstRunEpochs");
            }
            if (!(learningRate > 0.0f) || !Float.isFinite(learningRate)) {
                throw new IllegalArgumentException("learningRate must be positive and finite");
            }
        }

        public static Options defaults() {
            return builder().build();
        }

        public static Builder builder() {
            return new Builder();
        }

        public static final class Builder {
            private Path checkpointDir;
            private String device = "auto";
            private boolean failOnAcceleratorFallback;
            private int firstRunEpochs = 1;
            private int resumedEpochs = 2;
            private float learningRate = 0.01f;

            private Builder() {
            }

            public Builder checkpointDir(Path checkpointDir) {
                this.checkpointDir = checkpointDir;
                return this;
            }

            public Builder device(String device) {
                this.device = device;
                return this;
            }

            public Builder accelerator(String device) {
                return device(device);
            }

            public Builder failOnAcceleratorFallback(boolean failOnAcceleratorFallback) {
                this.failOnAcceleratorFallback = failOnAcceleratorFallback;
                return this;
            }

            public Builder requireAccelerator() {
                return failOnAcceleratorFallback(true);
            }

            public Builder firstRunEpochs(int firstRunEpochs) {
                this.firstRunEpochs = firstRunEpochs;
                return this;
            }

            public Builder resumedEpochs(int resumedEpochs) {
                this.resumedEpochs = resumedEpochs;
                return this;
            }

            public Builder learningRate(float learningRate) {
                this.learningRate = learningRate;
                return this;
            }

            public Options build() {
                return new Options(
                        checkpointDir,
                        device,
                        failOnAcceleratorFallback,
                        firstRunEpochs,
                        resumedEpochs,
                        learningRate);
            }
        }
    }

    public record Result(
            Path checkpointDir,
            TrainingSummary firstRun,
            TrainingSummary resumedRun,
            List<Check> checks) {
        public Result {
            checkpointDir = Objects.requireNonNull(checkpointDir, "checkpointDir must not be null")
                    .toAbsolutePath()
                    .normalize();
            firstRun = Objects.requireNonNull(firstRun, "firstRun must not be null");
            resumedRun = Objects.requireNonNull(resumedRun, "resumedRun must not be null");
            checks = List.copyOf(checks == null ? List.of() : checks);
        }

        public boolean passed() {
            return checks.stream().allMatch(Check::passed);
        }

        public List<Check> failures() {
            return checks.stream().filter(check -> !check.passed()).toList();
        }

        public Optional<Check> check(String name) {
            return checks.stream().filter(check -> check.name().equals(name)).findFirst();
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException("Trainer runtime smoke failed: " + failures());
            }
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("checkpointDir", checkpointDir.toString());
            map.put("passed", passed());
            map.put("checks", checks.stream().map(Check::toMap).toList());
            map.put("firstRun", summaryMap(firstRun));
            map.put("resumedRun", summaryMap(resumedRun));
            return Map.copyOf(map);
        }
    }

    public record Check(String name, boolean passed, String detail) {
        public Check {
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("name must not be blank");
            }
            detail = detail == null ? "" : detail;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", name);
            map.put("passed", passed);
            map.put("detail", detail);
            return Map.copyOf(map);
        }
    }

    private static List<Check> evaluate(
            Path checkpointDir,
            Options options,
            TrainingSummary firstRun,
            TrainingSummary resumedRun) {
        List<Check> checks = new ArrayList<>();
        TrainerCheckpointLayout layout = TrainerCheckpointLayout.from(checkpointDir);
        Map<String, Object> firstMetadata = firstRun.metadata();
        Map<String, Object> resumedMetadata = resumedRun.metadata();

        checks.add(check(
                "firstRunEpochCount",
                firstRun.epochCount() == options.firstRunEpochs(),
                "expected " + options.firstRunEpochs() + ", got " + firstRun.epochCount()));
        checks.add(check(
                "resumedRunEpochCount",
                resumedRun.epochCount() == options.resumedEpochs(),
                "expected " + options.resumedEpochs() + ", got " + resumedRun.epochCount()));
        checks.add(check(
                "firstRunLossFinite",
                finite(firstRun.latestTrainLoss()) && finite(firstRun.latestValidationLoss()),
                "train=" + firstRun.latestTrainLoss() + ", validation=" + firstRun.latestValidationLoss()));
        checks.add(check(
                "resumedRunLossFinite",
                finite(resumedRun.latestTrainLoss()) && finite(resumedRun.latestValidationLoss()),
                "train=" + resumedRun.latestTrainLoss() + ", validation=" + resumedRun.latestValidationLoss()));
        checks.add(check(
                "checkpointFilesPresent",
                regular(layout.runtime())
                        && regular(layout.model())
                        && regular(layout.modelMetadata())
                        && regular(layout.optimizer())
                        && regular(layout.history())
                        && regular(layout.report())
                        && regular(layout.manifest()),
                "checkpointDir=" + checkpointDir));
        checks.add(check(
                "firstRunCheckpointSaved",
                bool(firstMetadata, "modelCheckpointSaved")
                        && bool(firstMetadata, "modelCheckpointMetadataSaved")
                        && bool(firstMetadata, "optimizerCheckpointSaved")
                        && bool(firstMetadata, "trainingHistorySaved")
                        && bool(firstMetadata, "trainingReportSaved")
                        && bool(firstMetadata, "checkpointManifestSaved")
                        && bool(firstMetadata, "runtimeCheckpointPresent"),
                "metadata=" + selectedMetadata(firstMetadata, List.of(
                        "modelCheckpointSaved",
                        "modelCheckpointMetadataSaved",
                        "optimizerCheckpointSaved",
                        "trainingHistorySaved",
                        "trainingReportSaved",
                        "checkpointManifestSaved",
                        "runtimeCheckpointPresent"))));
        checks.add(check(
                "resumedFromCheckpoint",
                bool(resumedMetadata, "resumedFromCheckpoint")
                        && bool(resumedMetadata, "modelCheckpointLoaded")
                        && bool(resumedMetadata, "modelCheckpointMetadataLoaded")
                        && bool(resumedMetadata, "optimizerCheckpointLoaded")
                        && !bool(resumedMetadata, "checkpointResumeIssueDetected")
                        && !bool(resumedMetadata, "runtimeCheckpointIntegrityMismatch"),
                "metadata=" + selectedMetadata(resumedMetadata, List.of(
                        "resumedFromCheckpoint",
                        "modelCheckpointLoaded",
                        "modelCheckpointMetadataLoaded",
                        "optimizerCheckpointLoaded",
                        "checkpointResumeIssueDetected",
                        "runtimeCheckpointIntegrityMismatch"))));
        checks.add(check(
                "resumeProgressAdvanced",
                number(resumedMetadata, "optimizerStepCount") > number(firstMetadata, "optimizerStepCount"),
                "firstOptimizerSteps=" + number(firstMetadata, "optimizerStepCount")
                        + ", resumedOptimizerSteps=" + number(resumedMetadata, "optimizerStepCount")));
        checks.add(check(
                "deviceMetadataPresent",
                options.device().equals(text(firstMetadata, "requestedDevice"))
                        && options.device().equals(text(resumedMetadata, "requestedDevice"))
                        && !text(firstMetadata, "executionBackend").isBlank()
                        && !text(resumedMetadata, "executionBackend").isBlank(),
                "first=" + selectedMetadata(firstMetadata, List.of("requestedDevice", "executionBackend"))
                        + ", resumed="
                        + selectedMetadata(resumedMetadata, List.of("requestedDevice", "executionBackend"))));
        checks.add(check(
                "deviceFallbackPolicyRecorded",
                firstMetadata.containsKey("executionFallback")
                        && resumedMetadata.containsKey("executionFallback")
                        && firstMetadata.containsKey("requestedDeviceAvailable")
                        && resumedMetadata.containsKey("requestedDeviceAvailable"),
                "first=" + selectedMetadata(firstMetadata, List.of(
                        "executionFallback",
                        "requestedDeviceAvailable",
                        "executionAccelerated"))
                        + ", resumed="
                        + selectedMetadata(resumedMetadata, List.of(
                                "executionFallback",
                                "requestedDeviceAvailable",
                                "executionAccelerated"))));
        return checks;
    }

    private static Path resolvedCheckpointDir(Path checkpointDir) throws IOException {
        Path resolved = checkpointDir == null
                ? Files.createTempDirectory("gollek-trainer-runtime-smoke")
                : checkpointDir.toAbsolutePath().normalize();
        Files.createDirectories(resolved);
        return resolved;
    }

    private static List<Batch> regressionTrainBatches() {
        return List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f, 4f}, new float[] {6f, 8f}, 2));
    }

    private static List<Batch> regressionValidationBatches() {
        return List.of(batch(new float[] {1.5f, 2.5f}, new float[] {3f, 5f}, 2));
    }

    private static Batch batch(float[] inputs, float[] targets, int rows) {
        return new Batch(
                GradTensor.of(inputs, rows, 1),
                GradTensor.of(targets, rows, 1));
    }

    private static Check check(String name, boolean passed, String detail) {
        return new Check(name, passed, detail);
    }

    private static boolean finite(Double value) {
        return value != null && Double.isFinite(value);
    }

    private static boolean regular(Path path) {
        return path != null && Files.isRegularFile(path);
    }

    private static boolean bool(Map<String, Object> metadata, String key) {
        return Boolean.TRUE.equals(metadata.get(key));
    }

    private static int number(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        return value instanceof Number number ? number.intValue() : 0;
    }

    private static String text(Map<String, Object> metadata, String key) {
        Object value = metadata.get(key);
        return value == null ? "" : value.toString();
    }

    private static Map<String, Object> selectedMetadata(Map<String, Object> metadata, List<String> keys) {
        Map<String, Object> selected = new LinkedHashMap<>();
        for (String key : keys) {
            if (metadata.containsKey(key)) {
                selected.put(key, metadata.get(key));
            }
        }
        return Map.copyOf(selected);
    }

    private static Map<String, Object> summaryMap(TrainingSummary summary) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("epochCount", summary.epochCount());
        map.put("bestValidationLoss", summary.bestValidationLoss());
        map.put("bestValidationEpoch", summary.bestValidationEpoch());
        map.put("latestTrainLoss", summary.latestTrainLoss());
        map.put("latestValidationLoss", summary.latestValidationLoss());
        map.put("durationMs", summary.durationMs());
        map.put("metadata", selectedMetadata(summary.metadata(), List.of(
                "requestedDevice",
                "executionBackend",
                "executionAccelerated",
                "executionFallback",
                "resumedFromCheckpoint",
                "modelCheckpointLoaded",
                "optimizerCheckpointLoaded",
                "optimizerStepCount")));
        return Map.copyOf(map);
    }

    private static void appendMetadataRow(
            StringBuilder markdown,
            String label,
            Map<String, Object> firstMetadata,
            Map<String, Object> resumedMetadata,
            String key) {
        markdown.append("| ")
                .append(markdownCell(label))
                .append(" | ")
                .append(markdownCell(value(firstMetadata, key)))
                .append(" | ")
                .append(markdownCell(value(resumedMetadata, key)))
                .append(" |\n");
    }

    private static Object value(Map<String, Object> metadata, String key) {
        return metadata == null ? "unknown" : metadata.getOrDefault(key, "unknown");
    }

    private static String status(boolean passed) {
        return passed ? "PASS" : "FAIL";
    }

    private static String markdownCell(Object value) {
        return String.valueOf(value == null ? "" : value)
                .replace("|", "\\|")
                .replace("\r", " ")
                .replace("\n", " ");
    }

    private static String xml(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder escaped = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '&' -> escaped.append("&amp;");
                case '<' -> escaped.append("&lt;");
                case '>' -> escaped.append("&gt;");
                case '"' -> escaped.append("&quot;");
                case '\'' -> escaped.append("&apos;");
                default -> escaped.append(c);
            }
        }
        return escaped.toString();
    }
}
