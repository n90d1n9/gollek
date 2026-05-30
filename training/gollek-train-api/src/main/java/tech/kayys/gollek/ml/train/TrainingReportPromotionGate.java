package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One-call promotion gate for CI/CD pipelines.
 */
public final class TrainingReportPromotionGate {
    private TrainingReportPromotionGate() {
    }

    public record Request(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory,
            TrainingReportPromotionArtifacts.Options artifactOptions) {
        public Request {
            reportFiles = normalizeReportFiles(reportFiles);
            if (baselineName == null || baselineName.isBlank()) {
                throw new IllegalArgumentException("baselineName must not be blank");
            }
            baselineName = baselineName.trim();
            policy = policy == null ? TrainingReportPortfolio.PromotionPolicy.defaultPolicy() : policy;
            outputDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                    .toAbsolutePath()
                    .normalize();
            artifactOptions = artifactOptions == null
                    ? TrainingReportPromotionArtifacts.Options.defaults()
                    : artifactOptions;
        }

        public static Request of(
                Map<String, Path> reportFiles,
                String baselineName,
                TrainingReportPortfolio.PromotionPolicy policy,
                Path outputDirectory) {
            return new Request(
                    reportFiles,
                    baselineName,
                    policy,
                    outputDirectory,
                    TrainingReportPromotionArtifacts.Options.defaults());
        }

        public static Request withSeverity(
                Map<String, Path> reportFiles,
                String baselineName,
                TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
                Path outputDirectory) {
            Objects.requireNonNull(
                    maxAllowedDiagnosticSeverity,
                    "maxAllowedDiagnosticSeverity must not be null");
            return of(
                    reportFiles,
                    baselineName,
                    TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                            .withMaxCandidateDiagnosticSeverity(maxAllowedDiagnosticSeverity),
                    outputDirectory);
        }
    }

    public record Result(
            TrainingReportPortfolio.PromotionReview review,
            TrainingReportPromotionArtifacts.ArtifactBundle artifacts,
            TrainingReportPromotionArtifacts.ArtifactVerification verification) {
        public Result {
            review = Objects.requireNonNull(review, "review must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public TrainingReportPortfolio.PromotionDecision decision() {
            return review.decision();
        }

        public boolean promotable() {
            return decision().promotable();
        }

        public boolean passed() {
            return promotable() && verification.passed() && sourceVerification().passed();
        }

        public TrainingReportPromotionArtifacts.SourceVerification sourceVerification() {
            return verifyRecordedSourceReports(verification.inspection());
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (passed()) {
                return "Promotion gate passed: " + decision().message();
            }
            if (!verification.passed()) {
                return "Promotion gate failed because artifacts did not verify: " + verification.message();
            }
            TrainingReportPromotionArtifacts.SourceVerification sourceVerification = sourceVerification();
            if (!sourceVerification.passed()) {
                return "Promotion gate failed because source reports did not verify: " + sourceVerification.message();
            }
            return "Promotion gate failed: " + decision().message();
        }

        public String markdown() {
            return TrainingReportPromotionGateMarkdown.render(this);
        }

        public String junitXml() {
            return TrainingReportPromotionGateJUnitXml.render(this);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("message", message());
            map.put("review", review.toMap());
            map.put("decision", decision().toMap());
            map.put("artifacts", artifacts.toMap());
            map.put("verification", verification.toMap());
            map.put("sourceVerification", sourceVerification().toMap());
            return Map.copyOf(map);
        }
    }

    public static Result evaluate(Request request) throws IOException {
        Request resolvedRequest = Objects.requireNonNull(request, "request must not be null");
        TrainingReportPortfolio portfolio = TrainingReportPortfolio.fromFiles(resolvedRequest.reportFiles());
        TrainingReportPortfolio.PromotionReview review =
                portfolio.promotionReview(resolvedRequest.baselineName(), resolvedRequest.policy());
        TrainingReportPromotionArtifacts.ArtifactBundle artifacts = TrainingReportPromotionArtifacts.write(
                resolvedRequest.outputDirectory(),
                review,
                resolvedRequest.artifactOptions());
        TrainingReportPromotionArtifacts.ArtifactVerification verification =
                TrainingReportPromotionArtifacts.verify(artifacts);
        return new Result(review, artifacts, verification);
    }

    private static TrainingReportPromotionArtifacts.SourceVerification verifyRecordedSourceReports(
            TrainingReportPromotionArtifacts.ArtifactInspection inspection) {
        try {
            return TrainingReportPromotionArtifacts.verifySourceReports(inspection);
        } catch (IOException e) {
            List<TrainingReportPromotionArtifacts.SourceReport> reports =
                    TrainingReportPromotionArtifacts.sourceReports(inspection);
            return new TrainingReportPromotionArtifacts.SourceVerification(
                    inspection,
                    reports,
                    List.of("Unable to verify promotion source reports for "
                            + inspection.jsonFile() + ": " + e.getMessage()));
        }
    }

    public static Result evaluate(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory) throws IOException {
        return evaluate(Request.of(reportFiles, baselineName, policy, outputDirectory));
    }

    public static Result evaluate(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory) throws IOException {
        return evaluate(Request.withSeverity(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory));
    }

    private static Map<String, Path> normalizeReportFiles(Map<String, Path> reportFiles) {
        Objects.requireNonNull(reportFiles, "reportFiles must not be null");
        if (reportFiles.isEmpty()) {
            throw new IllegalArgumentException("reportFiles must not be empty");
        }
        Map<String, Path> normalized = new LinkedHashMap<>();
        for (Map.Entry<String, Path> entry : reportFiles.entrySet()) {
            String name = entry.getKey();
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("report name must not be blank");
            }
            Path path = Objects.requireNonNull(entry.getValue(), "report file path must not be null")
                    .toAbsolutePath()
                    .normalize();
            normalized.put(name.trim(), path);
        }
        return Map.copyOf(normalized);
    }
}
