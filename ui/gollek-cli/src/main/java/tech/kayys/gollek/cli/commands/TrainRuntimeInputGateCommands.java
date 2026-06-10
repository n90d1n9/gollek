package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.Callable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGate;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGateArtifacts;

public final class TrainRuntimeInputGateCommands {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private TrainRuntimeInputGateCommands() {
    }

    @Command(
            name = "runtime-input-gate",
            mixinStandardHelpOptions = true,
            description = "Validate runtime input-profile bottlenecks in a canonical trainer report")
    public static final class RuntimeInputGate implements Callable<Integer> {
        @Parameters(index = "0", description = "Canonical training report JSON file")
        Path reportFile;

        @Option(names = "--policy", defaultValue = "default",
                description = "Input gate policy: default, strict, or permissive (default: ${DEFAULT-VALUE})")
        String policyName;

        @Option(names = "--max-dominant-scope-percent",
                description = "Maximum allowed percent for the dominant input scope")
        Double maxDominantScopePercent;

        @Option(names = "--max-dominant-stage-percent",
                description = "Maximum allowed percent for the dominant stage inside an input scope")
        Double maxDominantStagePercent;

        @Option(names = "--max-train-validation-ratio",
                description = "Maximum allowed train/validation input total ratio")
        Double maxTrainValidationRatio;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = { "--output-dir", "--write-artifacts" },
                description = "Directory for runtime input gate JSON, Markdown, and JUnit XML artifacts")
        Path outputDirectory;

        @Option(names = "--no-fail",
                description = "Always exit 0 after gate evaluation; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path report = requireReportFile(reportFile);
                TrainingReportRuntimeInputProfileGate.Result result =
                        Gollek.DL.trainingReportRuntimeInputProfileGate(report, buildPolicy());
                TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts =
                        outputDirectory == null
                                ? null
                                : Gollek.DL.writeTrainingReportRuntimeInputProfileGateArtifacts(
                                        outputDirectory,
                                        result);

                print(result, artifacts);
                return result.passed() || noFail ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to run runtime input-profile gate: " + error.getMessage());
                return 1;
            }
        }

        private TrainingReportRuntimeInputProfileGate.Policy buildPolicy() {
            TrainingReportRuntimeInputProfileGate.Policy policy = switch (normalize(policyName)) {
                case "", "default" -> TrainingReportRuntimeInputProfileGate.Policy.defaults();
                case "strict" -> TrainingReportRuntimeInputProfileGate.Policy.strict();
                case "permissive", "advisory" -> TrainingReportRuntimeInputProfileGate.Policy.permissive();
                default -> throw new IllegalArgumentException("Unknown runtime input-profile gate policy: "
                        + policyName);
            };
            if (maxDominantScopePercent != null) {
                policy = policy.withMaxDominantScopePercent(maxDominantScopePercent);
            }
            if (maxDominantStagePercent != null) {
                policy = policy.withMaxDominantStagePercent(maxDominantStagePercent);
            }
            if (maxTrainValidationRatio != null) {
                policy = policy.withMaxTrainToValidationTotalRatio(maxTrainValidationRatio);
            }
            return policy;
        }

        private void print(
                TrainingReportRuntimeInputProfileGate.Result result,
                TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts) throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> TrainRuntimeInputGateFormatter.printGateSummary(reportFile, result, artifacts);
                case JSON -> System.out.println(JSON.writeValueAsString(
                        TrainRuntimeInputGateFormatter.resultPayload(result, artifacts)));
                case MARKDOWN -> System.out.print(result.markdown());
                case JUNIT -> System.out.print(result.junitXml());
            }
        }
    }

    @Command(
            name = "verify-runtime-input-gate-artifacts",
            mixinStandardHelpOptions = true,
            description = "Verify runtime input-profile gate artifacts and optional checksums")
    public static final class VerifyRuntimeInputGateArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing runtime input-profile gate artifacts")
        Path artifactDirectory;

        @Option(names = "--json-sha256", description = "Expected SHA-256 for runtime-input-profile-gate.json")
        String expectedJsonSha256;

        @Option(names = "--markdown-sha256", description = "Expected SHA-256 for runtime-input-profile-gate.md")
        String expectedMarkdownSha256;

        @Option(names = "--junit-xml-sha256", description = "Expected SHA-256 for runtime-input-profile-gate.junit.xml")
        String expectedJunitXmlSha256;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary or json (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification =
                        TrainingReportRuntimeInputProfileGateArtifacts.verify(
                                TrainingReportRuntimeInputProfileGateArtifacts.read(requireDirectory(artifactDirectory)),
                                expectedJsonSha256,
                                expectedMarkdownSha256,
                                expectedJunitXmlSha256);
                printVerification(
                        verification,
                        parseFormat(formatName),
                        "Runtime input-profile gate artifact verification");
                return verification.passed() || noFail ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to verify runtime input-profile gate artifacts: " + error.getMessage());
                return 1;
            }
        }
    }

    @Command(
            name = "refresh-runtime-input-gate-artifacts",
            mixinStandardHelpOptions = true,
            description = "Regenerate runtime input-profile gate Markdown/JUnit reports from JSON")
    public static final class RefreshRuntimeInputGateArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing runtime-input-profile-gate.json")
        Path artifactDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary or json (default: ${DEFAULT-VALUE})")
        String formatName;

        @Override
        public Integer call() {
            try {
                OutputFormat format = parseFormat(formatName);
                if (format != OutputFormat.SUMMARY && format != OutputFormat.JSON) {
                    throw new IllegalArgumentException(
                            "Runtime input-profile gate refresh only supports summary or json output");
                }
                TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts =
                        Gollek.DL.refreshTrainingReportRuntimeInputProfileGateArtifacts(
                                requireDirectory(artifactDirectory));
                TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportRuntimeInputProfileGateArtifacts(artifacts);
                if (format == OutputFormat.JSON) {
                    System.out.println(JSON.writeValueAsString(
                            TrainRuntimeInputGateFormatter.artifactPayload(artifacts, verification)));
                } else {
                    System.out.println("Runtime input-profile gate artifacts refreshed: "
                            + status(verification.passed()));
                    printVerification(verification, OutputFormat.SUMMARY, null);
                }
                return verification.passed() ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to refresh runtime input-profile gate artifacts: " + error.getMessage());
                return 1;
            }
        }
    }

    static void printVerification(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification,
            OutputFormat format,
            String heading) throws IOException {
        if (format == OutputFormat.JSON) {
            System.out.println(JSON.writeValueAsString(verification.toMap()));
            return;
        }
        if (format != OutputFormat.SUMMARY) {
            throw new IllegalArgumentException(
                    "Runtime input-profile gate artifact verification only supports summary or json output");
        }
        TrainRuntimeInputGateFormatter.printVerificationSummary(verification, heading);
    }

    private static OutputFormat parseFormat(String value) {
        return switch (normalize(value)) {
            case "", "summary" -> OutputFormat.SUMMARY;
            case "json" -> OutputFormat.JSON;
            case "markdown", "md" -> OutputFormat.MARKDOWN;
            case "junit", "junitxml", "junit-xml", "xml" -> OutputFormat.JUNIT;
            default -> throw new IllegalArgumentException("Unknown output format: " + value);
        };
    }

    enum OutputFormat {
        SUMMARY,
        JSON,
        MARKDOWN,
        JUNIT
    }

    private static Path requireReportFile(Path reportFile) {
        if (reportFile == null) {
            throw new IllegalArgumentException("report file is required");
        }
        Path report = reportFile.toAbsolutePath().normalize();
        if (!Files.isRegularFile(report)) {
            throw new IllegalArgumentException("report file not found: " + report);
        }
        return report;
    }

    private static Path requireDirectory(Path directory) {
        if (directory == null) {
            throw new IllegalArgumentException("artifact directory is required");
        }
        Path resolved = directory.toAbsolutePath().normalize();
        if (!Files.isDirectory(resolved)) {
            throw new IllegalArgumentException("artifact directory not found: " + resolved);
        }
        return resolved;
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static String status(boolean passed) {
        return passed ? "PASS" : "FAIL";
    }
}
