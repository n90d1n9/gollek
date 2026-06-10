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
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileBudgetGate;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileBudgetGateArtifacts;

public final class TrainRuntimeProfileBudgetGateCommands {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    private TrainRuntimeProfileBudgetGateCommands() {
    }

    @Command(
            name = "runtime-profile-budget-gate",
            mixinStandardHelpOptions = true,
            description = "Validate trainer runtime-profile budgets in a canonical trainer report")
    public static final class RuntimeProfileBudgetGate implements Callable<Integer> {
        @Parameters(index = "0", description = "Canonical training report JSON file")
        Path reportFile;

        @Option(names = "--policy", defaultValue = "default",
                description = "Budget policy: default, strict, or permissive (default: ${DEFAULT-VALUE})")
        String policyName;

        @Option(names = "--max-primary-group-percent",
                description = "Maximum allowed percent for the primary runtime group")
        Double maxPrimaryGroupPercent;

        @Option(names = "--max-primary-hotspot-percent",
                description = "Maximum allowed percent for the primary runtime hotspot")
        Double maxPrimaryHotspotPercent;

        @Option(names = "--max-primary-hotspot-ms",
                description = "Maximum allowed total milliseconds for the primary runtime hotspot")
        Double maxPrimaryHotspotMillis;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = { "--output-dir", "--write-artifacts" },
                description = "Directory for runtime profile budget gate JSON, Markdown, and JUnit XML artifacts")
        Path outputDirectory;

        @Option(names = "--no-fail",
                description = "Always exit 0 after gate evaluation; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path report = requireReportFile(reportFile);
                TrainingReportRuntimeProfileBudgetGate.Result result =
                        Gollek.DL.trainingReportRuntimeProfileBudgetGate(report, buildPolicy());
                TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle artifacts =
                        outputDirectory == null
                                ? null
                                : Gollek.DL.writeTrainingReportRuntimeProfileBudgetGateArtifacts(
                                        outputDirectory,
                                        result);

                print(result, artifacts);
                return result.passed() || noFail ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to run runtime profile budget gate: " + error.getMessage());
                return 1;
            }
        }

        private TrainingReportRuntimeProfileBudgetGate.Policy buildPolicy() {
            TrainingReportRuntimeProfileBudgetGate.Policy policy = switch (normalize(policyName)) {
                case "", "default" -> TrainingReportRuntimeProfileBudgetGate.Policy.defaults();
                case "strict" -> TrainingReportRuntimeProfileBudgetGate.Policy.strict();
                case "permissive", "advisory" -> TrainingReportRuntimeProfileBudgetGate.Policy.permissive();
                default -> throw new IllegalArgumentException("Unknown runtime profile budget policy: "
                        + policyName);
            };
            if (maxPrimaryGroupPercent != null) {
                policy = policy.withMaxPrimaryGroupPercent(maxPrimaryGroupPercent);
            }
            if (maxPrimaryHotspotPercent != null) {
                policy = policy.withMaxPrimaryHotspotPercent(maxPrimaryHotspotPercent);
            }
            if (maxPrimaryHotspotMillis != null) {
                policy = policy.withMaxPrimaryHotspotTotalMillis(maxPrimaryHotspotMillis);
            }
            return policy;
        }

        private void print(
                TrainingReportRuntimeProfileBudgetGate.Result result,
                TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle artifacts) throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> TrainRuntimeProfileBudgetGateFormatter.printGateSummary(reportFile, result, artifacts);
                case JSON -> System.out.println(JSON.writeValueAsString(
                        TrainRuntimeProfileBudgetGateFormatter.resultPayload(result, artifacts)));
                case MARKDOWN -> System.out.print(result.markdown());
                case JUNIT -> System.out.print(result.junitXml());
            }
        }
    }

    @Command(
            name = "verify-runtime-profile-budget-gate-artifacts",
            mixinStandardHelpOptions = true,
            description = "Verify runtime profile budget gate artifacts and optional checksums")
    public static final class VerifyRuntimeProfileBudgetGateArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing runtime profile budget gate artifacts")
        Path artifactDirectory;

        @Option(names = "--json-sha256", description = "Expected SHA-256 for runtime-profile-budget-gate.json")
        String expectedJsonSha256;

        @Option(names = "--markdown-sha256", description = "Expected SHA-256 for runtime-profile-budget-gate.md")
        String expectedMarkdownSha256;

        @Option(names = "--junit-xml-sha256", description = "Expected SHA-256 for runtime-profile-budget-gate.junit.xml")
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
                TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification verification =
                        TrainingReportRuntimeProfileBudgetGateArtifacts.verify(
                                TrainingReportRuntimeProfileBudgetGateArtifacts.read(requireDirectory(artifactDirectory)),
                                expectedJsonSha256,
                                expectedMarkdownSha256,
                                expectedJunitXmlSha256);
                printVerification(
                        verification,
                        parseFormat(formatName),
                        "Runtime profile budget gate artifact verification");
                return verification.passed() || noFail ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to verify runtime profile budget gate artifacts: " + error.getMessage());
                return 1;
            }
        }
    }

    @Command(
            name = "refresh-runtime-profile-budget-gate-artifacts",
            mixinStandardHelpOptions = true,
            description = "Regenerate runtime profile budget gate Markdown/JUnit reports from JSON")
    public static final class RefreshRuntimeProfileBudgetGateArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing runtime-profile-budget-gate.json")
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
                            "Runtime profile budget gate refresh only supports summary or json output");
                }
                TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle artifacts =
                        Gollek.DL.refreshTrainingReportRuntimeProfileBudgetGateArtifacts(
                                requireDirectory(artifactDirectory));
                TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportRuntimeProfileBudgetGateArtifacts(artifacts);
                if (format == OutputFormat.JSON) {
                    System.out.println(JSON.writeValueAsString(
                            TrainRuntimeProfileBudgetGateFormatter.artifactPayload(artifacts, verification)));
                } else {
                    TrainRuntimeProfileBudgetGateFormatter.printRefreshSummary(artifacts, verification);
                }
                return verification.passed() ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to refresh runtime profile budget gate artifacts: " + error.getMessage());
                return 1;
            }
        }
    }

    static void printVerification(
            TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification verification,
            OutputFormat format,
            String heading) throws IOException {
        if (format == OutputFormat.JSON) {
            System.out.println(JSON.writeValueAsString(verification.toMap()));
            return;
        }
        if (format != OutputFormat.SUMMARY) {
            throw new IllegalArgumentException(
                    "Runtime profile budget gate artifact verification only supports summary or json output");
        }
        TrainRuntimeProfileBudgetGateFormatter.printVerificationSummary(verification, heading);
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
}
