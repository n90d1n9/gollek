package tech.kayys.gollek.cli.commands;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Callable;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.train.TrainingReportActionPlan;
import tech.kayys.gollek.ml.train.TrainingReportComparison;
import tech.kayys.gollek.ml.train.TrainingReportComparisonArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportComparisonExport;
import tech.kayys.gollek.ml.train.TrainingReportDiagnostics;
import tech.kayys.gollek.ml.train.TrainingReportPortfolio;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactManifest;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactPackage;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifacts;
import tech.kayys.gollek.ml.train.TrainerRuntimeSmoke;
import tech.kayys.gollek.ml.train.TrainerRuntimeSmokeArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGate;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactManifest;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportValidationArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportValidationPolicy;

@Dependent
@Unremovable
@Command(
        name = "train",
        mixinStandardHelpOptions = true,
        description = "Training utilities and report automation",
        subcommands = {
                TrainCommand.ValidateReport.class,
                TrainCommand.VerifyValidationArtifacts.class,
                TrainCommand.RefreshValidationArtifacts.class,
                TrainCommand.ActionPlan.class,
                TrainCommand.CompareReports.class,
                TrainCommand.VerifyComparisonArtifacts.class,
                TrainCommand.RefreshComparisonArtifacts.class,
                TrainCommand.VerifyArtifacts.class,
                TrainCommand.RefreshArtifacts.class,
                TrainCommand.SmokeRuntime.class,
                TrainCommand.VerifySmokeRuntimeArtifacts.class,
                TrainCommand.RefreshSmokeRuntimeArtifacts.class,
                TrainCommand.PortfolioExport.class,
                TrainCommand.VerifyPortfolioArtifacts.class,
                TrainCommand.PortfolioPackage.class,
                TrainCommand.VerifyPortfolioPackage.class,
                TrainCommand.PromotionGate.class,
                TrainCommand.VerifyPromotionGateArtifacts.class,
                TrainCommand.PromotionPackage.class,
                TrainCommand.VerifyPromotionPackage.class
        })
public class TrainCommand implements Runnable {
    private static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    @Override
    public void run() {
        System.out.println("Usage: gollek train <command> [options]");
        System.out.println();
        System.out.println("Training commands:");
        System.out.println("  validate-report <report.json>       Validate a canonical trainer report");
        System.out.println("  verify-validation-artifacts <dir>   Verify validation artifact bundle integrity");
        System.out.println("  refresh-validation-artifacts <dir>  Regenerate validation Markdown/JUnit from JSON");
        System.out.println("  action-plan <report.json>           Print trainer diagnostics and remediation actions");
        System.out.println("  compare-reports --baseline file --candidate file Compare two trainer reports");
        System.out.println("  verify-comparison-artifacts <dir>   Verify comparison artifact integrity");
        System.out.println("  refresh-comparison-artifacts <dir>  Regenerate comparison Markdown/JUnit/CSV from JSON");
        System.out.println("  verify-artifacts <dir>              Auto-detect and verify trainer artifacts");
        System.out.println("  refresh-artifacts <dir>             Auto-detect and refresh trainer artifacts");
        System.out.println("  smoke-runtime                       Run tiny train/checkpoint/resume runtime smoke");
        System.out.println("  verify-smoke-runtime-artifacts <dir> Verify smoke runtime artifact integrity");
        System.out.println("  refresh-smoke-runtime-artifacts <dir> Regenerate smoke Markdown/JUnit from JSON");
        System.out.println("  portfolio-export --report name=file Write a multi-run leaderboard/comparison bundle");
        System.out.println("  verify-portfolio-artifacts <dir>    Verify portfolio export artifact integrity");
        System.out.println("  portfolio-package --report name=file Package portfolio export with manifest");
        System.out.println("  verify-portfolio-package <dir>      Verify portfolio package manifest and artifacts");
        System.out.println("  promotion-gate --report name=file   Decide if a candidate report can be promoted");
        System.out.println("  verify-promotion-gate-artifacts <dir> Verify promotion gate artifact integrity");
        System.out.println("  promotion-package --report name=file Package a promotion gate with audit evidence");
        System.out.println("  verify-promotion-package <dir>      Verify package manifest, artifacts, and snapshots");
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
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

    private enum OutputFormat {
        SUMMARY,
        JSON,
        MARKDOWN,
        JUNIT
    }

    @Command(
            name = "smoke-runtime",
            mixinStandardHelpOptions = true,
            description = "Run a tiny trainer checkpoint/resume smoke through the public Gollek.DL runtime")
    public static class SmokeRuntime implements Callable<Integer> {
        @Option(names = { "--checkpoint-dir", "--output-dir" },
                description = "Directory for smoke checkpoints; defaults to a temporary directory")
        Path checkpointDir;

        @Option(names = { "--device", "--accelerator" }, defaultValue = "auto",
                description = "Requested runtime device: auto, cpu, metal, cuda, etc. (default: ${DEFAULT-VALUE})")
        String device;

        @Option(names = "--require-accelerator",
                description = "Fail if the requested accelerator falls back to another backend")
        boolean requireAccelerator;

        @Option(names = "--first-epochs", defaultValue = "1",
                description = "Epochs for the initial smoke run (default: ${DEFAULT-VALUE})")
        int firstRunEpochs;

        @Option(names = "--resume-epochs", defaultValue = "2",
                description = "Total epochs after resume (default: ${DEFAULT-VALUE})")
        int resumedEpochs;

        @Option(names = "--learning-rate", defaultValue = "0.01",
                description = "Learning rate for the tiny regression smoke (default: ${DEFAULT-VALUE})")
        float learningRate;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--artifact-dir",
                description = "Write JSON, Markdown, and JUnit XML smoke artifacts to this directory")
        Path artifactDir;

        @Option(names = "--report-file",
                description = "Write the smoke result JSON to this file for CI/release artifacts")
        Path reportFile;

        @Option(names = "--markdown-file",
                description = "Write the smoke result Markdown to this file for human review")
        Path markdownFile;

        @Option(names = "--junit-file",
                description = "Write the smoke result JUnit XML to this file for CI/release artifacts")
        Path junitFile;

        @Option(names = "--no-fail",
                description = "Always exit 0 after smoke execution; useful for advisory diagnostics")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                TrainerRuntimeSmoke.Result result = Gollek.DL.trainerRuntimeSmoke(
                        TrainerRuntimeSmoke.Options.builder()
                                .checkpointDir(checkpointDir)
                                .device(device)
                                .failOnAcceleratorFallback(requireAccelerator)
                                .firstRunEpochs(firstRunEpochs)
                                .resumedEpochs(resumedEpochs)
                                .learningRate(learningRate)
                                .build());
                SmokeArtifacts artifacts = writeArtifactsIfRequested(result);
                print(result, artifacts);
                boolean artifactsPassed = artifacts.verification() == null || artifacts.verification().passed();
                return (result.passed() && artifactsPassed) || noFail ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to run trainer runtime smoke: " + error.getMessage());
                return 1;
            }
        }

        private SmokeArtifacts writeArtifactsIfRequested(TrainerRuntimeSmoke.Result result) throws IOException {
            Path resolvedReportFile = resolveArtifactFile(
                    reportFile,
                    TrainerRuntimeSmokeArtifacts.DEFAULT_JSON_FILE_NAME);
            Path resolvedMarkdownFile = resolveArtifactFile(
                    markdownFile,
                    TrainerRuntimeSmokeArtifacts.DEFAULT_MARKDOWN_FILE_NAME);
            Path resolvedJunitFile = resolveArtifactFile(
                    junitFile,
                    TrainerRuntimeSmokeArtifacts.DEFAULT_JUNIT_XML_FILE_NAME);
            if (resolvedReportFile == null && resolvedMarkdownFile == null && resolvedJunitFile == null) {
                return new SmokeArtifacts(null, null, null, null);
            }
            if (resolvedReportFile != null && resolvedMarkdownFile != null && resolvedJunitFile != null) {
                TrainerRuntimeSmokeArtifacts.ArtifactBundle bundle = TrainerRuntimeSmokeArtifacts.writeFiles(
                        resolvedReportFile,
                        resolvedMarkdownFile,
                        resolvedJunitFile,
                        result);
                return new SmokeArtifacts(
                        bundle.jsonFile(),
                        bundle.markdownFile(),
                        bundle.junitXmlFile(),
                        TrainerRuntimeSmokeArtifacts.verify(bundle));
            }
            if (resolvedReportFile != null) {
                writeStringAtomically(resolvedReportFile, JSON.writeValueAsString(result.toMap()) + "\n");
            }
            if (resolvedMarkdownFile != null) {
                writeStringAtomically(resolvedMarkdownFile, TrainerRuntimeSmoke.renderMarkdown(result));
            }
            if (resolvedJunitFile != null) {
                writeStringAtomically(resolvedJunitFile, TrainerRuntimeSmoke.renderJUnitXml(result));
            }
            return new SmokeArtifacts(resolvedReportFile, resolvedMarkdownFile, resolvedJunitFile, null);
        }

        private Path resolveArtifactFile(Path explicitFile, String defaultFileName) {
            if (explicitFile != null) {
                return explicitFile.toAbsolutePath().normalize();
            }
            if (artifactDir == null) {
                return null;
            }
            return artifactDir.toAbsolutePath().normalize().resolve(defaultFileName).normalize();
        }

        private void print(
                TrainerRuntimeSmoke.Result result,
                SmokeArtifacts artifacts) throws IOException {
            OutputFormat format = parseFormat(formatName);
            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(result.toMap()));
                return;
            }
            if (format == OutputFormat.MARKDOWN) {
                System.out.print(TrainerRuntimeSmoke.renderMarkdown(result));
                return;
            }
            if (format == OutputFormat.JUNIT) {
                System.out.print(TrainerRuntimeSmoke.renderJUnitXml(result));
                return;
            }
            if (format != OutputFormat.SUMMARY) {
                throw new IllegalArgumentException(
                        "Trainer runtime smoke only supports summary, json, markdown, or junit output");
            }
            printSummary(result, artifacts);
        }

        private static void printSummary(
                TrainerRuntimeSmoke.Result result,
                SmokeArtifacts artifacts) {
            Map<String, Object> firstMetadata = result.firstRun().metadata();
            Map<String, Object> resumedMetadata = result.resumedRun().metadata();
            System.out.println("Trainer runtime smoke: " + status(result.passed()));
            System.out.println("Checkpoint dir: " + result.checkpointDir());
            if (artifacts.reportFile() != null) {
                System.out.println("Smoke report: " + artifacts.reportFile());
            }
            if (artifacts.markdownFile() != null) {
                System.out.println("Smoke Markdown: " + artifacts.markdownFile());
            }
            if (artifacts.junitFile() != null) {
                System.out.println("Smoke JUnit XML: " + artifacts.junitFile());
            }
            if (artifacts.verification() != null) {
                System.out.println("Smoke artifacts verified: " + status(artifacts.verification().passed()));
                if (!artifacts.verification().passed()) {
                    for (String failure : artifacts.verification().failures()) {
                        System.out.println("- " + failure);
                    }
                }
            }
            System.out.println("First run epochs: " + result.firstRun().epochCount());
            System.out.println("Resumed run epochs: " + result.resumedRun().epochCount());
            System.out.println("Requested device: " + value(resumedMetadata, "requestedDevice"));
            System.out.println("Execution backend: " + value(resumedMetadata, "executionBackend"));
            System.out.println("Execution accelerated: " + value(resumedMetadata, "executionAccelerated"));
            System.out.println("Execution fallback: " + value(resumedMetadata, "executionFallback"));
            System.out.println("Checks:");
            for (TrainerRuntimeSmoke.Check check : result.checks()) {
                System.out.println("- " + check.name() + ": " + status(check.passed())
                        + (check.detail().isBlank() ? "" : " (" + check.detail() + ")"));
            }
            if (!result.passed()) {
                System.out.println("Failures:");
                for (TrainerRuntimeSmoke.Check failure : result.failures()) {
                    System.out.println("- " + failure.name() + ": " + failure.detail());
                }
            }
            System.out.println("First optimizer steps: " + value(firstMetadata, "optimizerStepCount"));
            System.out.println("Resumed optimizer steps: " + value(resumedMetadata, "optimizerStepCount"));
        }

        private static Object value(Map<String, Object> metadata, String key) {
            return metadata.getOrDefault(key, "unknown");
        }

        private record SmokeArtifacts(
                Path reportFile,
                Path markdownFile,
                Path junitFile,
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification) {
        }
    }

    @Command(
            name = "verify-smoke-runtime-artifacts",
            mixinStandardHelpOptions = true,
            description = "Verify trainer runtime smoke artifacts and optional checksums")
    public static class VerifySmokeRuntimeArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing trainer runtime smoke artifacts")
        Path artifactDirectory;

        @Option(names = "--json-sha256", description = "Expected SHA-256 for trainer-runtime-smoke.json")
        String expectedJsonSha256;

        @Option(names = "--markdown-sha256", description = "Expected SHA-256 for trainer-runtime-smoke.md")
        String expectedMarkdownSha256;

        @Option(names = "--junit-xml-sha256", description = "Expected SHA-256 for trainer-runtime-smoke.junit.xml")
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
                Path directory = requireDirectory(artifactDirectory);
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification =
                        TrainerRuntimeSmokeArtifacts.verify(
                                TrainerRuntimeSmokeArtifacts.read(directory),
                                expectedJsonSha256,
                                expectedMarkdownSha256,
                                expectedJunitXmlSha256);
                print(verification);
                return verification.passed() || noFail ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to verify trainer runtime smoke artifacts: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainerRuntimeSmokeArtifacts.ArtifactVerification verification) throws IOException {
            OutputFormat format = parseFormat(formatName);
            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(verification.toMap()));
                return;
            }
            if (format != OutputFormat.SUMMARY) {
                throw new IllegalArgumentException("Smoke artifact verification only supports summary or json output");
            }
            System.out.println("Trainer runtime smoke artifact verification: "
                    + status(verification.passed()));
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("Smoke passed: " + status(verification.inspection().smokePassed()));
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    @Command(
            name = "refresh-smoke-runtime-artifacts",
            mixinStandardHelpOptions = true,
            description = "Regenerate trainer runtime smoke Markdown/JUnit reports from JSON")
    public static class RefreshSmokeRuntimeArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing trainer-runtime-smoke.json")
        Path artifactDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary or json (default: ${DEFAULT-VALUE})")
        String formatName;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                OutputFormat format = parseFormat(formatName);
                if (format != OutputFormat.SUMMARY && format != OutputFormat.JSON) {
                    throw new IllegalArgumentException("Smoke artifact refresh only supports summary or json output");
                }
                TrainerRuntimeSmokeArtifacts.ArtifactBundle artifacts =
                        Gollek.DL.refreshTrainerRuntimeSmokeArtifacts(directory);
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainerRuntimeSmokeArtifacts(artifacts);
                print(artifacts, verification, format);
                return verification.passed() ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to refresh trainer runtime smoke artifacts: " + error.getMessage());
                return 1;
            }
        }

        private static void print(
                TrainerRuntimeSmokeArtifacts.ArtifactBundle artifacts,
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification,
                OutputFormat format) throws IOException {
            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(smokeRuntimePayload(artifacts, verification)));
            } else {
                printSummary(artifacts, verification);
            }
        }

        private static void printSummary(
                TrainerRuntimeSmokeArtifacts.ArtifactBundle artifacts,
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification) {
            System.out.println("Trainer runtime smoke artifacts refreshed: " + status(verification.passed()));
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Source JSON: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("Markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("JUnit XML: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
            System.out.println("Smoke passed: " + status(artifacts.smokePassed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static Map<String, Object> smokeRuntimePayload(
                TrainerRuntimeSmokeArtifacts.ArtifactBundle artifacts,
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "smoke-runtime");
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }
    }

    @Command(
            name = "validate-report",
            mixinStandardHelpOptions = true,
            description = "Validate a canonical trainer report and optionally write CI artifacts")
    public static class ValidateReport implements Callable<Integer> {
        @Parameters(index = "0", description = "Canonical training report JSON file")
        Path reportFile;

        @Option(names = "--policy", defaultValue = "strict",
                description = "Validation policy: strict or permissive (default: ${DEFAULT-VALUE})")
        String policyName;

        @Option(names = "--max-diagnostic-severity",
                description = "Maximum accepted diagnostic severity: INFO, WARNING, or CRITICAL")
        String maxDiagnosticSeverity;

        @Option(names = "--allow-run-health-failure",
                description = "Do not fail when the trainer run health gate failed")
        boolean allowRunHealthFailure;

        @Option(names = "--allow-stale-diagnostics",
                description = "Do not fail when diagnostics were backfilled or recomputed")
        boolean allowStaleDiagnostics;

        @Option(names = "--allow-missing-validation",
                description = "Do not fail when validation loss is missing")
        boolean allowMissingValidation;

        @Option(names = "--allow-checkpoint-integrity-failure",
                description = "Do not fail when checkpoint integrity metadata reports failures")
        boolean allowCheckpointIntegrityFailure;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = { "--output-dir", "--write-artifacts" },
                description = "Directory for validation JSON, Markdown, and JUnit XML artifacts")
        Path outputDirectory;

        @Option(names = "--no-fail",
                description = "Always exit 0 after validation; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path report = requireReportFile(reportFile);
                TrainingReportValidationPolicy policy = buildPolicy();
                TrainingReportValidationPolicy.Result result = Gollek.DL.validateTrainingReport(report, policy);
                TrainingReportValidationArtifacts.ArtifactBundle artifacts = writeArtifactsIfRequested(result);

                print(result, artifacts);
                if (result.failed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to validate training report: " + error.getMessage());
                return 1;
            }
        }

        private TrainingReportValidationPolicy buildPolicy() {
            TrainingReportValidationPolicy policy = switch (normalize(policyName)) {
                case "", "strict", "default" -> TrainingReportValidationPolicy.strict();
                case "permissive", "advisory" -> TrainingReportValidationPolicy.permissive();
                default -> throw new IllegalArgumentException(
                        "Unknown training report validation policy: " + policyName);
            };
            if (hasText(maxDiagnosticSeverity)) {
                policy = policy.withMaxDiagnosticSeverity(parseSeverity(maxDiagnosticSeverity));
            }
            if (allowRunHealthFailure) {
                policy = policy.withRequireRunHealthGate(false);
            }
            if (allowStaleDiagnostics) {
                policy = policy.withRequireFreshDiagnostics(false);
            }
            if (allowMissingValidation) {
                policy = policy.withRequireValidation(false);
            }
            if (allowCheckpointIntegrityFailure) {
                policy = policy.withRequireCheckpointIntegrity(false);
            }
            return policy;
        }

        private TrainingReportValidationArtifacts.ArtifactBundle writeArtifactsIfRequested(
                TrainingReportValidationPolicy.Result result) throws IOException {
            if (outputDirectory == null) {
                return null;
            }
            return Gollek.DL.writeTrainingReportValidationArtifacts(outputDirectory, result);
        }

        private void print(
                TrainingReportValidationPolicy.Result result,
                TrainingReportValidationArtifacts.ArtifactBundle artifacts) throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(result, artifacts);
                case JSON -> System.out.println(JSON.writeValueAsString(outputPayload(result, artifacts)));
                case MARKDOWN -> System.out.print(result.markdown());
                case JUNIT -> System.out.print(result.junitXml());
            }
        }

        private void printSummary(
                TrainingReportValidationPolicy.Result result,
                TrainingReportValidationArtifacts.ArtifactBundle artifacts) {
            System.out.println("Training report validation: " + (result.passed() ? "PASS" : "FAIL"));
            System.out.println("Report: " + reportFile.toAbsolutePath().normalize());
            System.out.println("Policy: " + policyLabel(result.policy()));
            System.out.println("Diagnostic threshold: " + result.policy().maxDiagnosticSeverity());
            System.out.println("Diagnostics provenance: " + result.diagnosticProvenance().status());
            System.out.println("Run health: " + result.runHealth().status());
            System.out.println("Validation loss available: " + result.validationAvailable());
            if (result.failed()) {
                System.out.println("Failure codes: " + String.join(", ", result.failureCodes()));
                for (String reason : result.reasons()) {
                    System.out.println("- " + reason);
                }
            } else {
                System.out.println(result.message());
            }
            if (artifacts != null) {
                System.out.println("Artifacts:");
                System.out.println("- json: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
                System.out.println("- markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
                System.out.println("- junitXml: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
            }
        }

        private static Map<String, Object> outputPayload(
                TrainingReportValidationPolicy.Result result,
                TrainingReportValidationArtifacts.ArtifactBundle artifacts) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("result", result.toMap());
            if (artifacts != null) {
                payload.put("artifacts", artifacts.toMap());
            }
            return payload;
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

        private static String policyLabel(TrainingReportValidationPolicy policy) {
            if (TrainingReportValidationPolicy.strict().equals(policy)) {
                return "strict";
            }
            if (TrainingReportValidationPolicy.permissive().equals(policy)) {
                return "permissive";
            }
            return policy.toMap().toString();
        }
    }

    @Command(
            name = "verify-validation-artifacts",
            mixinStandardHelpOptions = true,
            description = "Verify training report validation artifacts and optional checksums")
    public static class VerifyValidationArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing validation artifacts")
        Path artifactDirectory;

        @Option(names = "--json-sha256", description = "Expected SHA-256 for training-report-validation.json")
        String expectedJsonSha256;

        @Option(names = "--markdown-sha256", description = "Expected SHA-256 for training-report-validation.md")
        String expectedMarkdownSha256;

        @Option(names = "--junit-xml-sha256", description = "Expected SHA-256 for training-report-validation.junit.xml")
        String expectedJunitXmlSha256;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, or markdown (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportValidationArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportValidationArtifacts(
                                directory,
                                expectedJsonSha256,
                                expectedMarkdownSha256,
                                expectedJunitXmlSha256);

                print(verification);
                if (!verification.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to verify training report validation artifacts: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportValidationArtifacts.ArtifactVerification verification) throws IOException {
            OutputFormat format = parseFormat(formatName);
            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(verification.toMap()));
                return;
            }
            if (format != OutputFormat.SUMMARY) {
                throw new IllegalArgumentException("Artifact verification only supports summary or json output");
            }
            System.out.println("Training validation artifact verification: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }

    }

    @Command(
            name = "refresh-validation-artifacts",
            mixinStandardHelpOptions = true,
            description = "Regenerate derived validation artifacts from training-report-validation.json")
    public static class RefreshValidationArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing training-report-validation.json")
        Path artifactDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportValidationArtifacts.ArtifactBundle artifacts =
                        Gollek.DL.refreshTrainingReportValidationArtifacts(directory);
                TrainingReportValidationArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportValidationArtifacts(artifacts);

                print(artifacts, verification);
                return verification.passed() ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to refresh training validation artifacts: " + error.getMessage());
                return 1;
            }
        }

        private void print(
                TrainingReportValidationArtifacts.ArtifactBundle artifacts,
                TrainingReportValidationArtifacts.ArtifactVerification verification) throws IOException {
            OutputFormat format = parseFormat(formatName);
            switch (format) {
                case SUMMARY -> printSummary(artifacts, verification);
                case JSON -> System.out.println(JSON.writeValueAsString(outputPayload(artifacts, verification)));
                case MARKDOWN -> System.out.print(artifacts.result().markdown());
                case JUNIT -> System.out.print(artifacts.result().junitXml());
            }
        }

        private static void printSummary(
                TrainingReportValidationArtifacts.ArtifactBundle artifacts,
                TrainingReportValidationArtifacts.ArtifactVerification verification) {
            System.out.println("Training validation artifacts refreshed: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Source JSON: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("Markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("JUnit XML: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static Map<String, Object> outputPayload(
                TrainingReportValidationArtifacts.ArtifactBundle artifacts,
                TrainingReportValidationArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }
    }

    @Command(
            name = "action-plan",
            mixinStandardHelpOptions = true,
            description = "Print a diagnostics-driven action plan for one canonical training report")
    public static class ActionPlan implements Callable<Integer> {
        @Parameters(index = "0", description = "Canonical training report JSON file")
        Path reportFile;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, or markdown (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 even when the action plan requires attention")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path report = requireReportFile(reportFile);
                TrainingReportActionPlan actionPlan = Gollek.DL.trainingReportActionPlan(report);

                print(actionPlan);
                if (actionPlan.requiresAttention() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to build training action plan: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportActionPlan actionPlan) throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(actionPlan);
                case JSON -> System.out.println(JSON.writeValueAsString(actionPlan.toMap()));
                case MARKDOWN -> System.out.print(Gollek.DL.trainingReportActionPlanMarkdown(actionPlan));
                case JUNIT -> throw new IllegalArgumentException(
                        "Training action plan only supports summary, json, or markdown output");
            }
        }

        private void printSummary(TrainingReportActionPlan actionPlan) {
            System.out.println("Training action plan: " + actionPlan.status());
            System.out.println("Report: " + reportFile.toAbsolutePath().normalize());
            System.out.println("Highest diagnostic severity: "
                    + actionPlan.diagnosticSummary().highestSeverity());
            System.out.println("Diagnostics: " + actionPlan.diagnosticSummary().total());
            System.out.println("Recommendations: " + actionPlan.recommendations().size());
            System.out.println("Blockers: " + actionPlan.blockers().size());
            if (!actionPlan.actionItems().isEmpty()) {
                System.out.println("Action items:");
                for (String action : actionPlan.actionItems()) {
                    System.out.println("- " + action);
                }
            }
        }
    }

    @Command(
            name = "compare-reports",
            mixinStandardHelpOptions = true,
            description = "Compare baseline and candidate trainer reports for regressions")
    public static class CompareReports implements Callable<Integer> {
        @Option(names = "--baseline", required = true,
                description = "Baseline canonical training report JSON file")
        Path baselineReportFile;

        @Option(names = "--candidate", required = true,
                description = "Candidate canonical training report JSON file")
        Path candidateReportFile;

        @Option(names = "--max-finding-severity", defaultValue = "INFO",
                description = "Maximum accepted comparison finding severity: INFO, WARNING, or CRITICAL")
        String maxFindingSeverityName = "INFO";

        @Option(names = "--output-dir",
                description = "Optional directory for comparison JSON, Markdown, JUnit XML, and CSV artifacts")
        Path outputDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after comparison; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path baseline = requireReportFile(baselineReportFile);
                Path candidate = requireReportFile(candidateReportFile);
                TrainingReportDiagnostics.Severity maxFindingSeverity = parseSeverity(maxFindingSeverityName);
                TrainingReportComparison comparison = Gollek.DL.compareTrainingReports(baseline, candidate);
                TrainingReportComparisonExport export = Gollek.DL.trainingReportComparisonExport(comparison);
                TrainingReportDiagnostics.GateResult gate = comparison.gate(maxFindingSeverity);
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts = writeArtifactsIfRequested(export);

                print(comparison, export, gate, artifacts);
                if (!gate.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to compare training reports: " + error.getMessage());
                return 1;
            }
        }

        private TrainingReportComparisonArtifacts.ArtifactBundle writeArtifactsIfRequested(
                TrainingReportComparisonExport export) throws IOException {
            if (outputDirectory == null) {
                return null;
            }
            Path directory = requireOutputDirectory(outputDirectory);
            return Gollek.DL.writeTrainingReportComparisonArtifacts(directory, export);
        }

        private void print(
                TrainingReportComparison comparison,
                TrainingReportComparisonExport export,
                TrainingReportDiagnostics.GateResult gate,
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts) throws IOException {
            OutputFormat format = parseFormat(formatName);
            switch (format) {
                case SUMMARY -> printSummary(comparison, export, gate, artifacts);
                case JSON -> System.out.println(JSON.writeValueAsString(outputPayload(
                        comparison,
                        export,
                        gate,
                        artifacts)));
                case MARKDOWN -> System.out.print(Gollek.DL.trainingReportComparisonMarkdown(export));
                case JUNIT -> System.out.print(Gollek.DL.trainingReportComparisonJUnitXml(export));
            }
        }

        private void printSummary(
                TrainingReportComparison comparison,
                TrainingReportComparisonExport export,
                TrainingReportDiagnostics.GateResult gate,
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts) {
            System.out.println("Training report comparison: " + (gate.passed() ? "PASS" : "FAIL"));
            System.out.println("Baseline: " + baselineReportFile.toAbsolutePath().normalize());
            System.out.println("Candidate: " + candidateReportFile.toAbsolutePath().normalize());
            System.out.println("Metrics: " + export.metricCount());
            System.out.println("Findings: " + export.findingCount());
            System.out.println("Improved metrics: " + comparison.improvedMetrics().size());
            System.out.println("Regressed metrics: " + comparison.regressedMetrics().size());
            System.out.println(gate.message());
            if (!gate.passed()) {
                System.out.println("Failing findings:");
                for (TrainingReportDiagnostics.Finding finding : gate.failingFindings()) {
                    System.out.println("- " + finding.severity() + " " + finding.code()
                            + ": " + finding.message());
                }
            }
            if (artifacts != null) {
                System.out.println("Artifacts:");
                System.out.println("- json: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
                System.out.println("- markdown: " + artifacts.markdownFile()
                        + " sha256=" + artifacts.markdownSha256());
                System.out.println("- junitXml: " + artifacts.junitXmlFile()
                        + " sha256=" + artifacts.junitXmlSha256());
                System.out.println("- metricsCsv: " + artifacts.metricsCsvFile()
                        + " sha256=" + artifacts.metricsCsvSha256());
                System.out.println("- findingsCsv: " + artifacts.findingsCsvFile()
                        + " sha256=" + artifacts.findingsCsvSha256());
            }
        }

        private static Map<String, Object> outputPayload(
                TrainingReportComparison comparison,
                TrainingReportComparisonExport export,
                TrainingReportDiagnostics.GateResult gate,
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("comparison", comparison.toMap());
            payload.put("export", export.toMap());
            payload.put("gate", gatePayload(gate));
            if (artifacts != null) {
                payload.put("artifacts", artifacts.toMap());
            }
            return Map.copyOf(payload);
        }

        private static Map<String, Object> gatePayload(TrainingReportDiagnostics.GateResult gate) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("passed", gate.passed());
            payload.put("maxAllowedSeverity", gate.maxAllowedSeverity().name());
            payload.put("message", gate.message());
            payload.put("summary", gate.summary().toMap());
            payload.put("failingCodes", gate.failingCodes());
            payload.put("failingFindings", TrainingReportDiagnostics.toMaps(gate.failingFindings()));
            return Map.copyOf(payload);
        }

    }

    @Command(
            name = "verify-comparison-artifacts",
            mixinStandardHelpOptions = true,
            description = "Verify training report comparison artifacts and optional checksums")
    public static class VerifyComparisonArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing comparison artifacts")
        Path artifactDirectory;

        @Option(names = "--json-sha256", description = "Expected SHA-256 for comparison-export.json")
        String expectedJsonSha256;

        @Option(names = "--markdown-sha256", description = "Expected SHA-256 for comparison-report.md")
        String expectedMarkdownSha256;

        @Option(names = "--junit-xml-sha256", description = "Expected SHA-256 for comparison-report.junit.xml")
        String expectedJunitXmlSha256;

        @Option(names = "--metrics-csv-sha256", description = "Expected SHA-256 for comparison-metrics.csv")
        String expectedMetricsCsvSha256;

        @Option(names = "--findings-csv-sha256", description = "Expected SHA-256 for comparison-findings.csv")
        String expectedFindingsCsvSha256;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportComparisonArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportComparisonArtifacts(
                                directory,
                                expectedJsonSha256,
                                expectedMarkdownSha256,
                                expectedJunitXmlSha256,
                                expectedMetricsCsvSha256,
                                expectedFindingsCsvSha256);

                print(verification);
                if (!verification.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to verify training comparison artifacts: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportComparisonArtifacts.ArtifactVerification verification) throws IOException {
            OutputFormat format = parseFormat(formatName);
            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(verification.toMap()));
                return;
            }
            if (format == OutputFormat.MARKDOWN) {
                System.out.print(Gollek.DL.trainingReportComparisonArtifactVerificationMarkdown(verification));
                return;
            }
            if (format == OutputFormat.JUNIT) {
                System.out.print(Gollek.DL.trainingReportComparisonArtifactVerificationJUnitXml(verification));
                return;
            }
            if (format != OutputFormat.SUMMARY) {
                throw new IllegalArgumentException(
                        "Comparison artifact verification only supports summary, json, markdown, or junit output");
            }
            System.out.println("Training comparison artifact verification: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Metrics CSV checksum: " + status(verification.metricsCsvSha256Matches()));
            System.out.println("Findings CSV checksum: " + status(verification.findingsCsvSha256Matches()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            System.out.println("Metrics CSV matches JSON: " + status(verification.metricsCsvMatchesJson()));
            System.out.println("Findings CSV matches JSON: " + status(verification.findingsCsvMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    @Command(
            name = "refresh-comparison-artifacts",
            mixinStandardHelpOptions = true,
            description = "Regenerate derived comparison artifacts from comparison-export.json")
    public static class RefreshComparisonArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing comparison-export.json")
        Path artifactDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts =
                        Gollek.DL.refreshTrainingReportComparisonArtifacts(directory);
                TrainingReportComparisonArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportComparisonArtifacts(artifacts);

                print(artifacts, verification);
                return verification.passed() ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to refresh training comparison artifacts: " + error.getMessage());
                return 1;
            }
        }

        private void print(
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts,
                TrainingReportComparisonArtifacts.ArtifactVerification verification) throws IOException {
            OutputFormat format = parseFormat(formatName);
            switch (format) {
                case SUMMARY -> printSummary(artifacts, verification);
                case JSON -> System.out.println(JSON.writeValueAsString(outputPayload(artifacts, verification)));
                case MARKDOWN -> System.out.print(Gollek.DL.trainingReportComparisonArtifactMarkdown(artifacts));
                case JUNIT -> System.out.print(Gollek.DL.trainingReportComparisonArtifactVerificationJUnitXml(
                        verification));
            }
        }

        private static void printSummary(
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts,
                TrainingReportComparisonArtifacts.ArtifactVerification verification) {
            System.out.println("Training comparison artifacts refreshed: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Source JSON: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("Markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("JUnit XML: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
            System.out.println("Metrics CSV: " + artifacts.metricsCsvFile()
                    + " sha256=" + artifacts.metricsCsvSha256());
            System.out.println("Findings CSV: " + artifacts.findingsCsvFile()
                    + " sha256=" + artifacts.findingsCsvSha256());
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            System.out.println("Metrics CSV matches JSON: " + status(verification.metricsCsvMatchesJson()));
            System.out.println("Findings CSV matches JSON: " + status(verification.findingsCsvMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static Map<String, Object> outputPayload(
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts,
                TrainingReportComparisonArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }
    }

    @Command(
            name = "verify-artifacts",
            mixinStandardHelpOptions = true,
            description = "Auto-detect trainer artifact bundles and run the matching verifier")
    public static class VerifyArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing supported trainer artifacts")
        Path artifactDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary or json (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                OutputFormat format = parseFormat(formatName);
                if (format != OutputFormat.SUMMARY && format != OutputFormat.JSON) {
                    throw new IllegalArgumentException(
                            "Verify artifacts only supports summary or json output");
                }

                ArtifactKind kind = detectKind(directory);
                boolean passed = verify(directory, kind, format);
                return passed || noFail ? 0 : 2;
            } catch (Exception error) {
                System.err.println("Failed to verify trainer artifacts: " + error.getMessage());
                return 1;
            }
        }

        private boolean verify(Path directory, ArtifactKind kind, OutputFormat format) throws IOException {
            return switch (kind) {
                case VALIDATION -> verifyValidation(directory, format);
                case COMPARISON -> verifyComparison(directory, format);
                case PORTFOLIO_EXPORT -> verifyPortfolioExport(directory, format);
                case PORTFOLIO_PACKAGE -> verifyPortfolioPackage(directory, format);
                case PROMOTION_GATE -> verifyPromotionGate(directory, format);
                case PROMOTION_PACKAGE -> verifyPromotionPackage(directory, format);
                case SMOKE_RUNTIME -> verifySmokeRuntime(directory, format);
            };
        }

        private boolean verifyValidation(Path directory, OutputFormat format) throws IOException {
            TrainingReportValidationArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportValidationArtifacts(directory, null, null, null);
            if (format == OutputFormat.JSON) {
                printJson(ArtifactKind.VALIDATION, verification.toMap());
            } else {
                printValidationSummary(verification);
            }
            return verification.passed();
        }

        private boolean verifyComparison(Path directory, OutputFormat format) throws IOException {
            TrainingReportComparisonArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportComparisonArtifacts(
                            directory,
                            null,
                            null,
                            null,
                            null,
                            null);
            if (format == OutputFormat.JSON) {
                printJson(ArtifactKind.COMPARISON, verification.toMap());
            } else {
                printComparisonSummary(verification);
            }
            return verification.passed();
        }

        private boolean verifyPortfolioExport(Path directory, OutputFormat format) throws IOException {
            TrainingReportPortfolioArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(
                            directory,
                            null,
                            null,
                            null,
                            null,
                            null);
            if (format == OutputFormat.JSON) {
                printJson(ArtifactKind.PORTFOLIO_EXPORT, verification.toMap());
            } else {
                printPortfolioExportSummary(verification);
            }
            return verification.passed();
        }

        private boolean verifyPortfolioPackage(Path directory, OutputFormat format) throws IOException {
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification =
                    Gollek.DL.verifyTrainingReportPortfolioExportArtifactPackage(directory);
            if (format == OutputFormat.JSON) {
                printJson(ArtifactKind.PORTFOLIO_PACKAGE, verification.toMap());
            } else {
                printPortfolioPackageSummary(verification);
            }
            return verification.passed();
        }

        private boolean verifyPromotionGate(Path directory, OutputFormat format) throws IOException {
            TrainingReportPromotionGateArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportPromotionGateArtifacts(directory, null, null, null);
            if (format == OutputFormat.JSON) {
                printJson(ArtifactKind.PROMOTION_GATE, verification.toMap());
            } else {
                printPromotionGateSummary(verification);
            }
            return verification.passed();
        }

        private boolean verifyPromotionPackage(Path directory, OutputFormat format) throws IOException {
            TrainingReportPromotionGateArtifactPackage.PackageVerification verification =
                    Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(directory, null);
            if (format == OutputFormat.JSON) {
                printJson(ArtifactKind.PROMOTION_PACKAGE, verification.toMap());
            } else {
                printPromotionPackageSummary(verification);
            }
            return verification.passed();
        }

        private boolean verifySmokeRuntime(Path directory, OutputFormat format) throws IOException {
            TrainerRuntimeSmokeArtifacts.ArtifactVerification verification =
                    TrainerRuntimeSmokeArtifacts.verify(directory);
            if (format == OutputFormat.JSON) {
                printJson(ArtifactKind.SMOKE_RUNTIME, verification.toMap());
            } else {
                printSmokeRuntimeSummary(verification);
            }
            return verification.passed();
        }

        private static ArtifactKind detectKind(Path directory) {
            boolean portfolioPackage = hasFile(directory, TrainingReportPortfolioArtifactManifest.DEFAULT_FILE_NAME);
            boolean promotionPackage = hasFile(directory, TrainingReportPromotionGateArtifactManifest.DEFAULT_FILE_NAME);
            if (portfolioPackage && promotionPackage) {
                throw new IllegalArgumentException(
                        "Ambiguous trainer artifact directory: contains portfolio and promotion package manifests");
            }
            if (portfolioPackage) {
                return ArtifactKind.PORTFOLIO_PACKAGE;
            }
            if (promotionPackage) {
                return ArtifactKind.PROMOTION_PACKAGE;
            }

            List<ArtifactKind> matches = new ArrayList<>();
            if (hasFile(directory, TrainingReportValidationArtifacts.DEFAULT_JSON_FILE_NAME)) {
                matches.add(ArtifactKind.VALIDATION);
            }
            if (hasFile(directory, TrainingReportComparisonArtifacts.DEFAULT_JSON_FILE_NAME)) {
                matches.add(ArtifactKind.COMPARISON);
            }
            if (hasFile(directory, TrainingReportPortfolioArtifacts.DEFAULT_JSON_FILE_NAME)) {
                matches.add(ArtifactKind.PORTFOLIO_EXPORT);
            }
            if (hasFile(directory, TrainingReportPromotionGateArtifacts.DEFAULT_JSON_FILE_NAME)) {
                matches.add(ArtifactKind.PROMOTION_GATE);
            }
            if (hasFile(directory, TrainerRuntimeSmokeArtifacts.DEFAULT_JSON_FILE_NAME)) {
                matches.add(ArtifactKind.SMOKE_RUNTIME);
            }
            if (matches.size() == 1) {
                return matches.get(0);
            }
            if (matches.isEmpty()) {
                throw new IllegalArgumentException("Unsupported trainer artifact directory: expected one of "
                        + supportedArtifactMarkers());
            }
            throw new IllegalArgumentException("Ambiguous trainer artifact directory: matched " + matches);
        }

        private static boolean hasFile(Path directory, String fileName) {
            return Files.isRegularFile(directory.resolve(fileName));
        }

        private static void printJson(ArtifactKind kind, Map<String, Object> verification) throws IOException {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", kind.label());
            payload.put("verification", verification);
            System.out.println(JSON.writeValueAsString(Map.copyOf(payload)));
        }

        private static void printValidationSummary(
                TrainingReportValidationArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifact verification: " + status(verification.passed()));
            System.out.println("Type: " + ArtifactKind.VALIDATION.label());
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            printFailuresOrMessage(verification.passed(), verification.failures(), verification.message());
        }

        private static void printComparisonSummary(
                TrainingReportComparisonArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifact verification: " + status(verification.passed()));
            System.out.println("Type: " + ArtifactKind.COMPARISON.label());
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Metrics CSV checksum: " + status(verification.metricsCsvSha256Matches()));
            System.out.println("Findings CSV checksum: " + status(verification.findingsCsvSha256Matches()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            System.out.println("Metrics CSV matches JSON: " + status(verification.metricsCsvMatchesJson()));
            System.out.println("Findings CSV matches JSON: " + status(verification.findingsCsvMatchesJson()));
            printFailuresOrMessage(verification.passed(), verification.failures(), verification.message());
        }

        private static void printPortfolioExportSummary(
                TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifact verification: " + status(verification.passed()));
            System.out.println("Type: " + ArtifactKind.PORTFOLIO_EXPORT.label());
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("Leaderboard CSV checksum: " + status(verification.leaderboardCsvSha256Matches()));
            System.out.println("Comparison metrics CSV checksum: "
                    + status(verification.comparisonMetricsCsvSha256Matches()));
            System.out.println("Comparison findings CSV checksum: "
                    + status(verification.comparisonFindingsCsvSha256Matches()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("Leaderboard CSV matches JSON: " + status(verification.leaderboardCsvMatchesJson()));
            System.out.println("Comparison metrics CSV matches JSON: "
                    + status(verification.comparisonMetricsCsvMatchesJson()));
            System.out.println("Comparison findings CSV matches JSON: "
                    + status(verification.comparisonFindingsCsvMatchesJson()));
            printFailuresOrMessage(verification.passed(), verification.failures(), verification.message());
        }

        private static void printPortfolioPackageSummary(
                TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
            System.out.println("Training artifact verification: " + status(verification.passed()));
            System.out.println("Type: " + ArtifactKind.PORTFOLIO_PACKAGE.label());
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("Manifest checksum: " + status(verification.manifestSha256Matches()));
            System.out.println("Artifact bytes: " + status(verification.artifactBytesMatch()));
            System.out.println("Artifact sha256: " + status(verification.artifactSha256Match()));
            printFailuresOrMessage(verification.passed(), verification.failures(), verification.message());
        }

        private static void printPromotionGateSummary(
                TrainingReportPromotionGateArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifact verification: " + status(verification.passed()));
            System.out.println("Type: " + ArtifactKind.PROMOTION_GATE.label());
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            printFailuresOrMessage(verification.passed(), verification.failures(), verification.message());
        }

        private static void printPromotionPackageSummary(
                TrainingReportPromotionGateArtifactPackage.PackageVerification verification) {
            System.out.println("Training artifact verification: " + status(verification.passed()));
            System.out.println("Type: " + ArtifactKind.PROMOTION_PACKAGE.label());
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("Manifest checksum: "
                    + status(verification.manifestVerification().manifestSha256Matches()));
            System.out.println("Artifact bytes: "
                    + status(verification.manifestVerification().artifactBytesMatch()));
            System.out.println("Artifact sha256: "
                    + status(verification.manifestVerification().artifactSha256Match()));
            System.out.println("Source snapshots: "
                    + status(verification.sourceSnapshotVerification().passed()));
            printFailuresOrMessage(verification.passed(), verification.failures(), verification.message());
        }

        private static void printSmokeRuntimeSummary(
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifact verification: " + status(verification.passed()));
            System.out.println("Type: " + ArtifactKind.SMOKE_RUNTIME.label());
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("Smoke passed: " + status(verification.inspection().smokePassed()));
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            printFailuresOrMessage(verification.passed(), verification.failures(), verification.message());
        }

        private static void printFailuresOrMessage(boolean passed, List<String> failures, String message) {
            if (!passed) {
                for (String failure : failures) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(message);
            }
        }

        private static String supportedArtifactMarkers() {
            return String.join(
                    ", ",
                    TrainingReportValidationArtifacts.DEFAULT_JSON_FILE_NAME,
                    TrainingReportComparisonArtifacts.DEFAULT_JSON_FILE_NAME,
                    TrainingReportPortfolioArtifacts.DEFAULT_JSON_FILE_NAME,
                    TrainingReportPortfolioArtifactManifest.DEFAULT_FILE_NAME,
                    TrainingReportPromotionGateArtifacts.DEFAULT_JSON_FILE_NAME,
                    TrainingReportPromotionGateArtifactManifest.DEFAULT_FILE_NAME,
                    TrainerRuntimeSmokeArtifacts.DEFAULT_JSON_FILE_NAME);
        }

        private enum ArtifactKind {
            VALIDATION("validation"),
            COMPARISON("comparison"),
            PORTFOLIO_EXPORT("portfolio-export"),
            PORTFOLIO_PACKAGE("portfolio-package"),
            PROMOTION_GATE("promotion-gate"),
            PROMOTION_PACKAGE("promotion-package"),
            SMOKE_RUNTIME("smoke-runtime");

            private final String label;

            ArtifactKind(String label) {
                this.label = label;
            }

            String label() {
                return label;
            }
        }
    }

    @Command(
            name = "refresh-artifacts",
            mixinStandardHelpOptions = true,
            description = "Auto-detect repairable trainer artifacts and regenerate derived files")
    public static class RefreshArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing supported trainer artifact JSON")
        Path artifactDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary or json (default: ${DEFAULT-VALUE})")
        String formatName;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                OutputFormat format = parseFormat(formatName);
                if (format != OutputFormat.SUMMARY && format != OutputFormat.JSON) {
                    throw new IllegalArgumentException(
                            "Refresh artifacts only supports summary or json output");
                }

                boolean hasPortfolioPackageManifest = Files.exists(directory.resolve(
                        TrainingReportPortfolioArtifactManifest.DEFAULT_FILE_NAME));
                boolean hasPromotionPackageManifest = Files.exists(directory.resolve(
                        TrainingReportPromotionGateArtifactManifest.DEFAULT_FILE_NAME));
                if (hasPortfolioPackageManifest && hasPromotionPackageManifest) {
                    throw new IllegalArgumentException(
                            "Ambiguous trainer artifact directory: contains portfolio and promotion package manifests");
                }
                if (hasPromotionPackageManifest) {
                    return refreshPromotionPackage(directory, format);
                }
                if (hasPortfolioPackageManifest) {
                    return refreshPortfolioPackage(directory, format);
                }

                boolean hasValidationJson = Files.exists(directory.resolve(
                        TrainingReportValidationArtifacts.DEFAULT_JSON_FILE_NAME));
                boolean hasComparisonJson = Files.exists(directory.resolve(
                        TrainingReportComparisonArtifacts.DEFAULT_JSON_FILE_NAME));
                boolean hasPortfolioJson = Files.exists(directory.resolve(
                        TrainingReportPortfolioArtifacts.DEFAULT_JSON_FILE_NAME));
                boolean hasPromotionGateJson = Files.exists(directory.resolve(
                        TrainingReportPromotionGateArtifacts.DEFAULT_JSON_FILE_NAME));
                boolean hasSmokeRuntimeJson = Files.exists(directory.resolve(
                        TrainerRuntimeSmokeArtifacts.DEFAULT_JSON_FILE_NAME));

                int matches = (hasValidationJson ? 1 : 0)
                        + (hasComparisonJson ? 1 : 0)
                        + (hasPortfolioJson ? 1 : 0)
                        + (hasPromotionGateJson ? 1 : 0)
                        + (hasSmokeRuntimeJson ? 1 : 0);
                if (matches != 1) {
                    throw new IllegalArgumentException(matches > 1
                            ? "Ambiguous trainer artifact directory: contains multiple refreshable artifact JSON files"
                            : "Unsupported trainer artifact directory: expected "
                                    + TrainingReportValidationArtifacts.DEFAULT_JSON_FILE_NAME
                                    + " or "
                                    + TrainingReportComparisonArtifacts.DEFAULT_JSON_FILE_NAME
                                    + " or "
                                    + TrainingReportPortfolioArtifacts.DEFAULT_JSON_FILE_NAME
                                    + " or "
                                    + TrainingReportPromotionGateArtifacts.DEFAULT_JSON_FILE_NAME
                                    + " or "
                                    + TrainerRuntimeSmokeArtifacts.DEFAULT_JSON_FILE_NAME);
                }

                if (hasValidationJson) {
                    return refreshValidation(directory, format);
                }
                if (hasComparisonJson) {
                    return refreshComparison(directory, format);
                }
                if (hasPromotionGateJson) {
                    return refreshPromotionGate(directory, format);
                }
                if (hasSmokeRuntimeJson) {
                    return refreshSmokeRuntime(directory, format);
                }
                return refreshPortfolio(directory, format);
            } catch (Exception error) {
                System.err.println("Failed to refresh trainer artifacts: " + error.getMessage());
                return 1;
            }
        }

        private Integer refreshValidation(Path directory, OutputFormat format) throws IOException {
            TrainingReportValidationArtifacts.ArtifactBundle artifacts =
                    Gollek.DL.refreshTrainingReportValidationArtifacts(directory);
            TrainingReportValidationArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportValidationArtifacts(artifacts);

            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(validationPayload(artifacts, verification)));
            } else {
                printValidationSummary(artifacts, verification);
            }
            return verification.passed() ? 0 : 2;
        }

        private Integer refreshPortfolio(Path directory, OutputFormat format) throws IOException {
            TrainingReportPortfolioArtifacts.ArtifactBundle artifacts =
                    Gollek.DL.refreshTrainingReportPortfolioExportArtifacts(directory);
            TrainingReportPortfolioArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(artifacts);

            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(portfolioPayload(artifacts, verification)));
            } else {
                printPortfolioSummary(artifacts, verification);
            }
            return verification.passed() ? 0 : 2;
        }

        private Integer refreshPortfolioPackage(Path directory, OutputFormat format) throws IOException {
            TrainingReportPortfolioArtifactPackage.PackageRefresh refresh =
                    Gollek.DL.refreshTrainingReportPortfolioExportArtifactPackage(directory);

            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(portfolioPackagePayload(refresh)));
            } else {
                printPortfolioPackageSummary(refresh);
            }
            return refresh.passed() ? 0 : 2;
        }

        private Integer refreshPromotionPackage(Path directory, OutputFormat format) throws IOException {
            TrainingReportPromotionGateArtifactPackage.PackageRefresh refresh =
                    Gollek.DL.refreshTrainingReportPromotionGateArtifactPackage(directory);

            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(promotionPackagePayload(refresh)));
            } else {
                printPromotionPackageSummary(refresh);
            }
            return refresh.passed() ? 0 : 2;
        }

        private Integer refreshPromotionGate(Path directory, OutputFormat format) throws IOException {
            TrainingReportPromotionGateArtifacts.ArtifactInspection inspection =
                    Gollek.DL.refreshTrainingReportPromotionGateArtifacts(directory);
            TrainingReportPromotionGateArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportPromotionGateArtifacts(directory, null, null, null);

            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(promotionGatePayload(inspection, verification)));
            } else {
                printPromotionGateSummary(inspection, verification);
            }
            return verification.passed() ? 0 : 2;
        }

        private Integer refreshComparison(Path directory, OutputFormat format) throws IOException {
            TrainingReportComparisonArtifacts.ArtifactBundle artifacts =
                    Gollek.DL.refreshTrainingReportComparisonArtifacts(directory);
            TrainingReportComparisonArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainingReportComparisonArtifacts(artifacts);

            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(comparisonPayload(artifacts, verification)));
            } else {
                printComparisonSummary(artifacts, verification);
            }
            return verification.passed() ? 0 : 2;
        }

        private Integer refreshSmokeRuntime(Path directory, OutputFormat format) throws IOException {
            TrainerRuntimeSmokeArtifacts.ArtifactBundle artifacts =
                    Gollek.DL.refreshTrainerRuntimeSmokeArtifacts(directory);
            TrainerRuntimeSmokeArtifacts.ArtifactVerification verification =
                    Gollek.DL.verifyTrainerRuntimeSmokeArtifacts(artifacts);

            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(smokeRuntimePayload(artifacts, verification)));
            } else {
                printSmokeRuntimeSummary(artifacts, verification);
            }
            return verification.passed() ? 0 : 2;
        }

        private static void printValidationSummary(
                TrainingReportValidationArtifacts.ArtifactBundle artifacts,
                TrainingReportValidationArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifacts refreshed: " + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Type: validation");
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Source JSON: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("Markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("JUnit XML: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static void printComparisonSummary(
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts,
                TrainingReportComparisonArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifacts refreshed: " + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Type: comparison");
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Source JSON: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("Markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("JUnit XML: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
            System.out.println("Metrics CSV: " + artifacts.metricsCsvFile()
                    + " sha256=" + artifacts.metricsCsvSha256());
            System.out.println("Findings CSV: " + artifacts.findingsCsvFile()
                    + " sha256=" + artifacts.findingsCsvSha256());
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            System.out.println("Metrics CSV matches JSON: " + status(verification.metricsCsvMatchesJson()));
            System.out.println("Findings CSV matches JSON: " + status(verification.findingsCsvMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static void printPortfolioSummary(
                TrainingReportPortfolioArtifacts.ArtifactBundle artifacts,
                TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifacts refreshed: " + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Type: portfolio-export");
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Source JSON: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("Markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("Leaderboard CSV: " + artifacts.leaderboardCsvFile()
                    + " sha256=" + artifacts.leaderboardCsvSha256());
            System.out.println("Comparison metrics CSV: " + artifacts.comparisonMetricsCsvFile()
                    + " sha256=" + artifacts.comparisonMetricsCsvSha256());
            System.out.println("Comparison findings CSV: " + artifacts.comparisonFindingsCsvFile()
                    + " sha256=" + artifacts.comparisonFindingsCsvSha256());
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("Leaderboard CSV matches JSON: " + status(verification.leaderboardCsvMatchesJson()));
            System.out.println("Comparison metrics CSV matches JSON: "
                    + status(verification.comparisonMetricsCsvMatchesJson()));
            System.out.println("Comparison findings CSV matches JSON: "
                    + status(verification.comparisonFindingsCsvMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static void printPortfolioPackageSummary(
                TrainingReportPortfolioArtifactPackage.PackageRefresh refresh) {
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification = refresh.verification();
            System.out.println("Training artifacts refreshed: " + (refresh.passed() ? "PASS" : "FAIL"));
            System.out.println("Type: portfolio-package");
            System.out.println("Directory: " + refresh.directory());
            System.out.println("Manifest: " + refresh.manifest().manifestFile()
                    + " sha256=" + refresh.manifest().manifestSha256());
            System.out.println("Markdown: " + refresh.artifacts().markdownFile()
                    + " sha256=" + refresh.artifacts().markdownSha256());
            System.out.println("Leaderboard CSV: " + refresh.artifacts().leaderboardCsvFile()
                    + " sha256=" + refresh.artifacts().leaderboardCsvSha256());
            System.out.println("Comparison metrics CSV: " + refresh.artifacts().comparisonMetricsCsvFile()
                    + " sha256=" + refresh.artifacts().comparisonMetricsCsvSha256());
            System.out.println("Comparison findings CSV: " + refresh.artifacts().comparisonFindingsCsvFile()
                    + " sha256=" + refresh.artifacts().comparisonFindingsCsvSha256());
            System.out.println("Verification report: " + refresh.reports().jsonFile()
                    + " sha256=" + refresh.reports().jsonSha256());
            System.out.println("Verification report Markdown: " + refresh.reports().markdownFile()
                    + " sha256=" + refresh.reports().markdownSha256());
            System.out.println("Verification report JUnit XML: " + refresh.reports().junitXmlFile()
                    + " sha256=" + refresh.reports().junitXmlSha256());
            System.out.println("Verification report artifacts: " + status(refresh.reportVerification().passed()));
            System.out.println("Artifact sha256: " + status(verification.artifactSha256Match()));
            verification.artifactVerificationOptional().ifPresent(artifactVerification -> {
                System.out.println("Markdown matches JSON: " + status(artifactVerification.markdownMatchesJson()));
                System.out.println("Leaderboard CSV matches JSON: "
                        + status(artifactVerification.leaderboardCsvMatchesJson()));
                System.out.println("Comparison metrics CSV matches JSON: "
                        + status(artifactVerification.comparisonMetricsCsvMatchesJson()));
                System.out.println("Comparison findings CSV matches JSON: "
                        + status(artifactVerification.comparisonFindingsCsvMatchesJson()));
            });
            if (!refresh.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static void printPromotionGateSummary(
                TrainingReportPromotionGateArtifacts.ArtifactInspection inspection,
                TrainingReportPromotionGateArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifacts refreshed: " + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Type: promotion-gate");
            System.out.println("Directory: " + inspection.directory());
            System.out.println("Source JSON: " + inspection.jsonFile() + " sha256=" + inspection.jsonSha256());
            System.out.println("Markdown: " + inspection.markdownFile() + " sha256=" + inspection.markdownSha256());
            System.out.println("JUnit XML: " + inspection.junitXmlFile() + " sha256=" + inspection.junitXmlSha256());
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static void printSmokeRuntimeSummary(
                TrainerRuntimeSmokeArtifacts.ArtifactBundle artifacts,
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification) {
            System.out.println("Training artifacts refreshed: " + status(verification.passed()));
            System.out.println("Type: smoke-runtime");
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Source JSON: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("Markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("JUnit XML: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
            System.out.println("Smoke passed: " + status(artifacts.smokePassed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static void printPromotionPackageSummary(
                TrainingReportPromotionGateArtifactPackage.PackageRefresh refresh) {
            TrainingReportPromotionGateArtifactPackage.PackageVerification verification = refresh.verification();
            System.out.println("Training artifacts refreshed: " + (refresh.passed() ? "PASS" : "FAIL"));
            System.out.println("Type: promotion-package");
            System.out.println("Directory: " + refresh.directory());
            System.out.println("Manifest: " + refresh.manifest().manifestFile()
                    + " sha256=" + refresh.manifest().manifestSha256());
            System.out.println("Markdown: " + refresh.artifacts().markdownFile()
                    + " sha256=" + refresh.artifacts().markdownSha256());
            System.out.println("JUnit XML: " + refresh.artifacts().junitXmlFile()
                    + " sha256=" + refresh.artifacts().junitXmlSha256());
            System.out.println("Verification report: " + refresh.reports().json().reportFile()
                    + " sha256=" + refresh.reports().json().reportSha256());
            System.out.println("Verification report artifacts: "
                    + status(refresh.reportVerification().passed()));
            System.out.println("Artifact sha256: "
                    + status(verification.manifestVerification().artifactSha256Match()));
            System.out.println("Markdown matches JSON: " + status(verification.manifestVerification()
                    .artifactVerification()
                    .markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.manifestVerification()
                    .artifactVerification()
                    .junitXmlMatchesJson()));
            System.out.println("Source snapshots: " + status(verification.sourceSnapshotVerification().passed()));
            if (!refresh.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static Map<String, Object> validationPayload(
                TrainingReportValidationArtifacts.ArtifactBundle artifacts,
                TrainingReportValidationArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "validation");
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }

        private static Map<String, Object> comparisonPayload(
                TrainingReportComparisonArtifacts.ArtifactBundle artifacts,
                TrainingReportComparisonArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "comparison");
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }

        private static Map<String, Object> portfolioPayload(
                TrainingReportPortfolioArtifacts.ArtifactBundle artifacts,
                TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "portfolio-export");
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }

        private static Map<String, Object> portfolioPackagePayload(
                TrainingReportPortfolioArtifactPackage.PackageRefresh refresh) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "portfolio-package");
            payload.put("refresh", refresh.toMap());
            return Map.copyOf(payload);
        }

        private static Map<String, Object> promotionPackagePayload(
                TrainingReportPromotionGateArtifactPackage.PackageRefresh refresh) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "promotion-package");
            payload.put("refresh", refresh.toMap());
            return Map.copyOf(payload);
        }

        private static Map<String, Object> promotionGatePayload(
                TrainingReportPromotionGateArtifacts.ArtifactInspection inspection,
                TrainingReportPromotionGateArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "promotion-gate");
            payload.put("inspection", inspection.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }

        private static Map<String, Object> smokeRuntimePayload(
                TrainerRuntimeSmokeArtifacts.ArtifactBundle artifacts,
                TrainerRuntimeSmokeArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("type", "smoke-runtime");
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }
    }

    @Command(
            name = "portfolio-export",
            mixinStandardHelpOptions = true,
            description = "Write a multi-run training report leaderboard and optional baseline comparisons")
    public static class PortfolioExport implements Callable<Integer> {
        @Option(names = "--report", required = true,
                description = "Named report in name=path form. Repeat for every run to include.")
        List<String> reportSpecs = new ArrayList<>();

        @Option(names = "--baseline",
                description = "Optional report name used for candidate comparison tables")
        String baselineName;

        @Option(names = "--output-dir", required = true,
                description = "Directory for portfolio JSON, Markdown, and CSV artifacts")
        Path outputDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, or markdown (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after artifact verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Map<String, Path> reports = parseReportSpecs(reportSpecs, 1);
                Path output = requireOutputDirectory(outputDirectory);
                TrainingReportPortfolioArtifacts.ArtifactBundle artifacts = hasText(baselineName)
                        ? Gollek.DL.writeTrainingReportPortfolioExportArtifacts(
                                output,
                                reports,
                                requireName(baselineName, "baseline"))
                        : Gollek.DL.writeTrainingReportPortfolioExportArtifacts(output, reports);
                TrainingReportPortfolioArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(artifacts);

                print(artifacts, verification);
                if (!verification.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to export training report portfolio: " + error.getMessage());
                return 1;
            }
        }

        private void print(
                TrainingReportPortfolioArtifacts.ArtifactBundle artifacts,
                TrainingReportPortfolioArtifacts.ArtifactVerification verification) throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(artifacts, verification);
                case JSON -> System.out.println(JSON.writeValueAsString(outputPayload(artifacts, verification)));
                case MARKDOWN -> System.out.print(Gollek.DL.trainingReportPortfolioArtifactMarkdown(artifacts));
                case JUNIT -> throw new IllegalArgumentException(
                        "Portfolio export only supports summary, json, or markdown output");
            }
        }

        private void printSummary(
                TrainingReportPortfolioArtifacts.ArtifactBundle artifacts,
                TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
            System.out.println("Training portfolio export: " + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + artifacts.directory());
            System.out.println("Entries: " + artifacts.export().entryCount());
            System.out.println("Comparison metrics: " + artifacts.export().comparisonMetricCount());
            System.out.println("Comparison findings: " + artifacts.export().comparisonFindingCount());
            System.out.println("Artifacts verified: " + status(verification.passed()));
            System.out.println("Artifacts:");
            System.out.println("- json: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("- markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("- leaderboardCsv: " + artifacts.leaderboardCsvFile()
                    + " sha256=" + artifacts.leaderboardCsvSha256());
            System.out.println("- comparisonMetricsCsv: " + artifacts.comparisonMetricsCsvFile()
                    + " sha256=" + artifacts.comparisonMetricsCsvSha256());
            System.out.println("- comparisonFindingsCsv: " + artifacts.comparisonFindingsCsvFile()
                    + " sha256=" + artifacts.comparisonFindingsCsvSha256());
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            }
        }

        private static Map<String, Object> outputPayload(
                TrainingReportPortfolioArtifacts.ArtifactBundle artifacts,
                TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("artifacts", artifacts.toMap());
            payload.put("verification", verification.toMap());
            return Map.copyOf(payload);
        }
    }

    @Command(
            name = "verify-portfolio-artifacts",
            mixinStandardHelpOptions = true,
            description = "Verify portfolio export artifacts and optional checksums")
    public static class VerifyPortfolioArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing portfolio export artifacts")
        Path artifactDirectory;

        @Option(names = "--json-sha256", description = "Expected SHA-256 for portfolio-export.json")
        String expectedJsonSha256;

        @Option(names = "--markdown-sha256", description = "Expected SHA-256 for portfolio-export.md")
        String expectedMarkdownSha256;

        @Option(names = "--leaderboard-csv-sha256",
                description = "Expected SHA-256 for portfolio-leaderboard.csv")
        String expectedLeaderboardCsvSha256;

        @Option(names = "--comparison-metrics-csv-sha256",
                description = "Expected SHA-256 for portfolio-comparison-metrics.csv")
        String expectedComparisonMetricsCsvSha256;

        @Option(names = "--comparison-findings-csv-sha256",
                description = "Expected SHA-256 for portfolio-comparison-findings.csv")
        String expectedComparisonFindingsCsvSha256;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, or markdown (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportPortfolioArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportPortfolioExportArtifacts(
                                directory,
                                expectedJsonSha256,
                                expectedMarkdownSha256,
                                expectedLeaderboardCsvSha256,
                                expectedComparisonMetricsCsvSha256,
                                expectedComparisonFindingsCsvSha256);

                print(verification);
                if (!verification.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to verify portfolio artifacts: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportPortfolioArtifacts.ArtifactVerification verification) throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(verification);
                case JSON -> System.out.println(JSON.writeValueAsString(verification.toMap()));
                case MARKDOWN -> System.out.print(Gollek.DL.trainingReportPortfolioArtifactVerificationMarkdown(
                        verification));
                case JUNIT -> throw new IllegalArgumentException(
                        "Portfolio artifact verification only supports summary, json, or markdown output");
            }
        }

        private void printSummary(TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
            System.out.println("Training portfolio artifact verification: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("Leaderboard CSV checksum: " + status(verification.leaderboardCsvSha256Matches()));
            System.out.println("Comparison metrics CSV checksum: "
                    + status(verification.comparisonMetricsCsvSha256Matches()));
            System.out.println("Comparison findings CSV checksum: "
                    + status(verification.comparisonFindingsCsvSha256Matches()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("Leaderboard CSV matches JSON: " + status(verification.leaderboardCsvMatchesJson()));
            System.out.println("Comparison metrics CSV matches JSON: "
                    + status(verification.comparisonMetricsCsvMatchesJson()));
            System.out.println("Comparison findings CSV matches JSON: "
                    + status(verification.comparisonFindingsCsvMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    @Command(
            name = "portfolio-package",
            mixinStandardHelpOptions = true,
            description = "Write and verify a manifest-backed portfolio export package")
    public static class PortfolioPackage implements Callable<Integer> {
        @Option(names = "--report", required = true,
                description = "Named report in name=path form. Repeat for every run to include.")
        List<String> reportSpecs = new ArrayList<>();

        @Option(names = "--baseline",
                description = "Optional report name used for candidate comparison tables")
        String baselineName;

        @Option(names = "--output-dir", required = true,
                description = "Directory for portfolio artifacts and manifest")
        Path outputDirectory;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after package verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Map<String, Path> reports = parseReportSpecs(reportSpecs, 1);
                Path output = requireOutputDirectory(outputDirectory);
                TrainingReportPortfolioArtifactPackage.VerifiedPackageBundle verified = hasText(baselineName)
                        ? Gollek.DL.writeAndVerifyTrainingReportPortfolioExportArtifactPackage(
                                output,
                                reports,
                                requireName(baselineName, "baseline"))
                        : Gollek.DL.writeAndVerifyTrainingReportPortfolioExportArtifactPackage(output, reports);

                print(verified);
                if (!verified.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to package training report portfolio: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportPortfolioArtifactPackage.VerifiedPackageBundle verified)
                throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(verified);
                case JSON -> System.out.println(JSON.writeValueAsString(verified.toMap()));
                case MARKDOWN -> System.out.print(Gollek.DL.trainingReportPortfolioArtifactPackageVerificationMarkdown(
                        verified.verification()));
                case JUNIT -> System.out.print(Gollek.DL.trainingReportPortfolioArtifactPackageVerificationJunitXml(
                        verified.verification()));
            }
        }

        private void printSummary(TrainingReportPortfolioArtifactPackage.VerifiedPackageBundle verified) {
            TrainingReportPortfolioArtifactPackage.PackageBundle bundle = verified.bundle();
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification = verified.verification();
            System.out.println("Training portfolio package: " + (verified.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + bundle.directory());
            System.out.println("Entries: " + bundle.entryCount());
            System.out.println("Comparison metrics: " + bundle.comparisonMetricCount());
            System.out.println("Comparison findings: " + bundle.comparisonFindingCount());
            System.out.println("Manifest: " + bundle.manifest().manifestFile()
                    + " sha256=" + bundle.manifest().manifestSha256());
            System.out.println("Manifest checksum: " + status(verification.manifestSha256Matches()));
            System.out.println("Artifact bytes: " + status(verification.artifactBytesMatch()));
            System.out.println("Artifact sha256: " + status(verification.artifactSha256Match()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    @Command(
            name = "verify-portfolio-package",
            mixinStandardHelpOptions = true,
            description = "Verify a portfolio package manifest and artifacts")
    public static class VerifyPortfolioPackage implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing the portfolio package")
        Path artifactDirectory;

        @Option(names = "--manifest-sha256",
                description = "Expected SHA-256 for portfolio-export.manifest.properties")
        String expectedManifestSha256;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportPortfolioArtifactManifest.ManifestVerification verification =
                        Gollek.DL.verifyTrainingReportPortfolioExportArtifactPackage(
                                directory,
                                expectedManifestSha256);

                print(verification);
                if (!verification.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to verify portfolio package: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportPortfolioArtifactManifest.ManifestVerification verification)
                throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(verification);
                case JSON -> System.out.println(JSON.writeValueAsString(verification.toMap()));
                case MARKDOWN -> System.out.print(Gollek.DL.trainingReportPortfolioArtifactPackageVerificationMarkdown(
                        verification));
                case JUNIT -> System.out.print(Gollek.DL.trainingReportPortfolioArtifactPackageVerificationJunitXml(
                        verification));
            }
        }

        private void printSummary(TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
            System.out.println("Training portfolio package verification: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("Manifest checksum: " + status(verification.manifestSha256Matches()));
            System.out.println("Artifact bytes: " + status(verification.artifactBytesMatch()));
            System.out.println("Artifact sha256: " + status(verification.artifactSha256Match()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    @Command(
            name = "promotion-gate",
            mixinStandardHelpOptions = true,
            description = "Run a CI promotion gate across named canonical trainer reports")
    public static class PromotionGate implements Callable<Integer> {
        @Option(names = "--report", required = true,
                description = "Named report in name=path form. Repeat for baseline and candidates.")
        List<String> reportSpecs = new ArrayList<>();

        @Option(names = "--baseline", required = true,
                description = "Report name to use as the current promoted baseline")
        String baselineName;

        @Option(names = "--output-dir", required = true,
                description = "Directory for promotion gate JSON, Markdown, and JUnit XML artifacts")
        Path outputDirectory;

        @Option(names = "--max-candidate-diagnostic-severity",
                description = "Maximum accepted candidate diagnostic severity: INFO, WARNING, or CRITICAL")
        String maxCandidateDiagnosticSeverity;

        @Option(names = "--max-comparison-finding-severity",
                description = "Maximum accepted comparison finding severity: INFO, WARNING, or CRITICAL")
        String maxComparisonFindingSeverity;

        @Option(names = "--minimum-validation-improvement",
                description = "Minimum validation-loss improvement required over the baseline")
        Double minimumValidationImprovement;

        @Option(names = "--allow-no-tracked-metric-improvement",
                description = "Do not require at least one tracked metric to improve")
        boolean allowNoTrackedMetricImprovement;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after gate evaluation; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Map<String, Path> reports = parseReportSpecs(reportSpecs);
                String baseline = requireName(baselineName, "baseline");
                Path output = requireOutputDirectory(outputDirectory);
                TrainingReportPortfolio.PromotionPolicy policy = buildPolicy();
                TrainingReportPromotionGate.Result result = Gollek.DL.runTrainingReportPromotionGate(
                        reports,
                        baseline,
                        policy,
                        output);
                TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts =
                        Gollek.DL.writeTrainingReportPromotionGateArtifacts(output, result);

                print(result, artifacts);
                if (!result.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to run training promotion gate: " + error.getMessage());
                return 1;
            }
        }

        private TrainingReportPortfolio.PromotionPolicy buildPolicy() {
            return promotionPolicy(
                    maxCandidateDiagnosticSeverity,
                    maxComparisonFindingSeverity,
                    minimumValidationImprovement,
                    allowNoTrackedMetricImprovement);
        }

        private void print(
                TrainingReportPromotionGate.Result result,
                TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts) throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(result, artifacts);
                case JSON -> System.out.println(JSON.writeValueAsString(outputPayload(result, artifacts)));
                case MARKDOWN -> System.out.print(Files.readString(artifacts.markdownFile()));
                case JUNIT -> System.out.print(Files.readString(artifacts.junitXmlFile()));
            }
        }

        private void printSummary(
                TrainingReportPromotionGate.Result result,
                TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts) {
            TrainingReportPortfolio.PromotionDecision decision = result.decision();
            System.out.println("Training promotion gate: " + (result.passed() ? "PASS" : "FAIL"));
            System.out.println("Status: " + decision.status());
            System.out.println("Baseline: " + decision.baseline().name());
            System.out.println("Candidate: " + decision.candidate()
                    .map(TrainingReportPortfolio.Entry::name)
                    .orElse("(none)"));
            System.out.println("Promotable: " + result.promotable());
            System.out.println("Artifacts verified: " + status(result.verification().passed()));
            System.out.println("Source reports verified: " + status(result.sourceVerification().passed()));
            System.out.println(result.message());
            if (!decision.reasons().isEmpty()) {
                System.out.println("Reasons:");
                for (String reason : decision.reasons()) {
                    System.out.println("- " + reason);
                }
            }
            System.out.println("Artifacts:");
            System.out.println("- json: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
            System.out.println("- markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
            System.out.println("- junitXml: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
        }

        private static Map<String, Object> outputPayload(
                TrainingReportPromotionGate.Result result,
                TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("result", result.toMap());
            payload.put("artifacts", artifacts.toMap());
            return Map.copyOf(payload);
        }
    }

    @Command(
            name = "verify-promotion-gate-artifacts",
            mixinStandardHelpOptions = true,
            description = "Verify promotion gate artifacts and optional checksums")
    public static class VerifyPromotionGateArtifacts implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing promotion gate artifacts")
        Path artifactDirectory;

        @Option(names = "--json-sha256", description = "Expected SHA-256 for promotion-gate.json")
        String expectedJsonSha256;

        @Option(names = "--markdown-sha256", description = "Expected SHA-256 for promotion-gate.md")
        String expectedMarkdownSha256;

        @Option(names = "--junit-xml-sha256", description = "Expected SHA-256 for promotion-gate.junit.xml")
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
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportPromotionGateArtifacts.ArtifactVerification verification =
                        Gollek.DL.verifyTrainingReportPromotionGateArtifacts(
                                directory,
                                expectedJsonSha256,
                                expectedMarkdownSha256,
                                expectedJunitXmlSha256);

                print(verification);
                if (!verification.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to verify promotion gate artifacts: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportPromotionGateArtifacts.ArtifactVerification verification) throws IOException {
            OutputFormat format = parseFormat(formatName);
            if (format == OutputFormat.JSON) {
                System.out.println(JSON.writeValueAsString(verification.toMap()));
                return;
            }
            if (format != OutputFormat.SUMMARY) {
                throw new IllegalArgumentException("Promotion gate artifact verification only supports summary or json output");
            }
            System.out.println("Promotion gate artifact verification: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
            System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
            System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
            System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
            System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
            System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    @Command(
            name = "promotion-package",
            mixinStandardHelpOptions = true,
            description = "Run, package, and verify a promotion gate with audit evidence")
    public static class PromotionPackage implements Callable<Integer> {
        @Option(names = "--report", required = true,
                description = "Named report in name=path form. Repeat for baseline and candidates.")
        List<String> reportSpecs = new ArrayList<>();

        @Option(names = "--baseline", required = true,
                description = "Report name to use as the current promoted baseline")
        String baselineName;

        @Option(names = "--output-dir", required = true,
                description = "Directory for package artifacts, manifest, verification reports, and evidence")
        Path outputDirectory;

        @Option(names = "--max-candidate-diagnostic-severity",
                description = "Maximum accepted candidate diagnostic severity: INFO, WARNING, or CRITICAL")
        String maxCandidateDiagnosticSeverity;

        @Option(names = "--max-comparison-finding-severity",
                description = "Maximum accepted comparison finding severity: INFO, WARNING, or CRITICAL")
        String maxComparisonFindingSeverity;

        @Option(names = "--minimum-validation-improvement",
                description = "Minimum validation-loss improvement required over the baseline")
        Double minimumValidationImprovement;

        @Option(names = "--allow-no-tracked-metric-improvement",
                description = "Do not require at least one tracked metric to improve")
        boolean allowNoTrackedMetricImprovement;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after package evaluation; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Map<String, Path> reports = parseReportSpecs(reportSpecs);
                String baseline = requireName(baselineName, "baseline");
                Path output = requireOutputDirectory(outputDirectory);
                TrainingReportPortfolio.PromotionPolicy policy = promotionPolicy(
                        maxCandidateDiagnosticSeverity,
                        maxComparisonFindingSeverity,
                        minimumValidationImprovement,
                        allowNoTrackedMetricImprovement);
                TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified =
                        Gollek.DL.runAndVerifyTrainingReportPromotionGateArtifactPackage(
                                reports,
                                baseline,
                                policy,
                                output);

                print(verified);
                if (!verified.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to run training promotion package: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified)
                throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(verified);
                case JSON -> System.out.println(JSON.writeValueAsString(verified.toMap()));
                case MARKDOWN -> System.out.print(
                        Gollek.DL.renderTrainingReportPromotionGatePackageVerificationMarkdown(
                                verified.verification()));
                case JUNIT -> System.out.print(
                        Gollek.DL.renderTrainingReportPromotionGatePackageVerificationJUnitXml(
                                verified.verification()));
            }
        }

        private void printSummary(TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verified) {
            TrainingReportPromotionGateArtifactPackage.PackageBundle pkg = verified.packageBundle();
            TrainingReportPromotionGateArtifactPackage.PackageVerification verification = verified.verification();
            System.out.println("Training promotion package: " + (verified.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + verified.directory());
            System.out.println("Promotable: " + verified.promotable());
            System.out.println("Package verification: " + status(verification.passed()));
            System.out.println("Source snapshots: " + status(verification.sourceSnapshotVerification().passed()));
            System.out.println("Manifest: " + pkg.manifest().manifestFile()
                    + " sha256=" + pkg.manifest().manifestSha256());
            System.out.println("Verification report: " + verified.reports().json().reportFile()
                    + " sha256=" + verified.reports().json().reportSha256());
            if (verified.index() != null) {
                System.out.println("Verification index: " + verified.index().indexFile()
                        + " sha256=" + verified.index().indexSha256());
            }
            if (verified.evidence() != null) {
                System.out.println("Evidence: " + verified.evidence().evidenceFile()
                        + " sha256=" + verified.evidence().evidenceSha256());
            }
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    @Command(
            name = "verify-promotion-package",
            mixinStandardHelpOptions = true,
            description = "Verify a promotion gate package manifest, artifacts, and source-report snapshots")
    public static class VerifyPromotionPackage implements Callable<Integer> {
        @Parameters(index = "0", description = "Directory containing the promotion package")
        Path artifactDirectory;

        @Option(names = "--manifest-sha256",
                description = "Expected SHA-256 for promotion-gate.manifest.properties")
        String expectedManifestSha256;

        @Option(names = "--format", defaultValue = "summary",
                description = "Output format: summary, json, markdown, or junit (default: ${DEFAULT-VALUE})")
        String formatName;

        @Option(names = "--no-fail",
                description = "Always exit 0 after verification; useful for advisory CI jobs")
        boolean noFail;

        @Override
        public Integer call() {
            try {
                Path directory = requireDirectory(artifactDirectory);
                TrainingReportPromotionGateArtifactPackage.PackageVerification verification =
                        Gollek.DL.verifyCompleteTrainingReportPromotionGateArtifactPackage(
                                directory,
                                expectedManifestSha256);

                print(verification);
                if (!verification.passed() && !noFail) {
                    return 2;
                }
                return 0;
            } catch (Exception error) {
                System.err.println("Failed to verify promotion package: " + error.getMessage());
                return 1;
            }
        }

        private void print(TrainingReportPromotionGateArtifactPackage.PackageVerification verification)
                throws IOException {
            switch (parseFormat(formatName)) {
                case SUMMARY -> printSummary(verification);
                case JSON -> System.out.println(JSON.writeValueAsString(verification.toMap()));
                case MARKDOWN -> System.out.print(
                        Gollek.DL.renderTrainingReportPromotionGatePackageVerificationMarkdown(verification));
                case JUNIT -> System.out.print(
                        Gollek.DL.renderTrainingReportPromotionGatePackageVerificationJUnitXml(verification));
            }
        }

        private void printSummary(TrainingReportPromotionGateArtifactPackage.PackageVerification verification) {
            System.out.println("Training promotion package verification: "
                    + (verification.passed() ? "PASS" : "FAIL"));
            System.out.println("Directory: " + verification.inspection().directory());
            System.out.println("Manifest checksum: "
                    + status(verification.manifestVerification().manifestSha256Matches()));
            System.out.println("Artifact bytes: "
                    + status(verification.manifestVerification().artifactBytesMatch()));
            System.out.println("Artifact sha256: "
                    + status(verification.manifestVerification().artifactSha256Match()));
            System.out.println("Source snapshots: "
                    + status(verification.sourceSnapshotVerification().passed()));
            if (!verification.passed()) {
                for (String failure : verification.failures()) {
                    System.out.println("- " + failure);
                }
            } else {
                System.out.println(verification.message());
            }
        }
    }

    private static TrainingReportDiagnostics.Severity parseSeverity(String value) {
        String normalized = normalize(value);
        for (TrainingReportDiagnostics.Severity severity : TrainingReportDiagnostics.Severity.values()) {
            if (severity.name().equals(normalized.toUpperCase(Locale.ROOT))) {
                return severity;
            }
        }
        throw new IllegalArgumentException("Unknown diagnostic severity: " + value);
    }

    private static TrainingReportPortfolio.PromotionPolicy promotionPolicy(
            String maxCandidateDiagnosticSeverity,
            String maxComparisonFindingSeverity,
            Double minimumValidationImprovement,
            boolean allowNoTrackedMetricImprovement) {
        TrainingReportPortfolio.PromotionPolicy policy = TrainingReportPortfolio.PromotionPolicy.defaultPolicy();
        if (hasText(maxCandidateDiagnosticSeverity)) {
            policy = policy.withMaxCandidateDiagnosticSeverity(parseSeverity(maxCandidateDiagnosticSeverity));
        }
        if (hasText(maxComparisonFindingSeverity)) {
            policy = policy.withMaxComparisonFindingSeverity(parseSeverity(maxComparisonFindingSeverity));
        }
        if (minimumValidationImprovement != null) {
            policy = policy.withMinimumValidationImprovement(minimumValidationImprovement.doubleValue());
        }
        if (allowNoTrackedMetricImprovement) {
            policy = policy.withRequireTrackedMetricImprovement(false);
        }
        return policy;
    }

    private static Map<String, Path> parseReportSpecs(List<String> reportSpecs) {
        return parseReportSpecs(reportSpecs, 2);
    }

    private static Map<String, Path> parseReportSpecs(List<String> reportSpecs, int minimumReports) {
        List<String> specs = reportSpecs == null ? List.of() : reportSpecs;
        if (specs.size() < minimumReports) {
            throw new IllegalArgumentException(
                    "at least " + minimumReports + " --report name=path entries are required");
        }
        Map<String, Path> reports = new LinkedHashMap<>();
        for (String spec : specs) {
            int separator = spec == null ? -1 : spec.indexOf('=');
            if (separator <= 0 || separator == spec.length() - 1) {
                throw new IllegalArgumentException("report must use name=path form: " + spec);
            }
            String name = requireName(spec.substring(0, separator), "report name");
            Path report = Path.of(spec.substring(separator + 1)).toAbsolutePath().normalize();
            if (!Files.isRegularFile(report)) {
                throw new IllegalArgumentException("report file not found for " + name + ": " + report);
            }
            if (reports.putIfAbsent(name, report) != null) {
                throw new IllegalArgumentException("duplicate report name: " + name);
            }
        }
        return reports;
    }

    private static String requireName(String value, String fieldName) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return value.trim();
    }

    private static Path requireOutputDirectory(Path directory) {
        if (directory == null) {
            throw new IllegalArgumentException("output directory is required");
        }
        return directory.toAbsolutePath().normalize();
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

    private static void writeStringAtomically(Path target, String content) throws IOException {
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path tempFile = target.resolveSibling(target.getFileName() + ".tmp");
        try {
            Files.writeString(tempFile, content, StandardCharsets.UTF_8);
            moveIntoPlace(tempFile, target);
        } catch (IOException | RuntimeException error) {
            Files.deleteIfExists(tempFile);
            throw error;
        }
    }

    private static void moveIntoPlace(Path tempFile, Path target) throws IOException {
        try {
            Files.move(tempFile, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException atomicMoveUnsupported) {
            Files.move(tempFile, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String status(boolean passed) {
        return passed ? "PASS" : "FAIL";
    }
}
