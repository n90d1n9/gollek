package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Profile-aware single-report validation gate for CI/CD pipelines.
 */
public final class TrainingReportQualityProfileValidationGate {
    private TrainingReportQualityProfileValidationGate() {
    }

    public record Request(
            Path reportFile,
            TrainingReportQualityProfile profile,
            Path outputDirectory,
            TrainingReportValidationArtifacts.Options artifactOptions) {
        public Request {
            reportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            profile = profile == null ? TrainingReportQualityProfile.strictCi() : profile;
            outputDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                    .toAbsolutePath()
                    .normalize();
            artifactOptions = artifactOptions == null
                    ? TrainingReportValidationArtifacts.Options.defaults()
                    : artifactOptions;
        }

        public static Request of(
                Path reportFile,
                TrainingReportQualityProfile profile,
                Path outputDirectory) {
            return new Request(
                    reportFile,
                    profile,
                    outputDirectory,
                    TrainingReportValidationArtifacts.Options.defaults());
        }

        public static Request ofProfile(
                Path reportFile,
                String profileId,
                Path outputDirectory) {
            return of(reportFile, TrainingReportQualityProfile.require(profileId), outputDirectory);
        }

        public static Request ofCatalogProfile(
                Path reportFile,
                TrainingReportQualityProfileCatalog catalog,
                String profileId,
                Path outputDirectory) {
            TrainingReportQualityProfileCatalog resolvedCatalog = catalog == null
                    ? TrainingReportQualityProfileCatalog.defaults()
                    : catalog;
            return of(reportFile, resolvedCatalog.require(profileId), outputDirectory);
        }

        public static Request ofCatalogProfile(
                Path reportFile,
                Path catalogJsonFile,
                String profileId,
                Path outputDirectory) throws IOException {
            return ofCatalogProfile(
                    reportFile,
                    TrainingReportQualityProfileCatalog.readJson(catalogJsonFile),
                    profileId,
                    outputDirectory);
        }
    }

    public record Result(
            TrainingReportQualityProfile profile,
            TrainingReportValidationPolicy.Result validation,
            TrainingReportValidationArtifacts.ArtifactBundle artifacts,
            TrainingReportValidationArtifacts.ArtifactVerification verification,
            TrainingReportPerformanceGate.Result performance,
            TrainingReportPerformanceGateArtifacts.ArtifactBundle performanceArtifacts,
            TrainingReportPerformanceGateArtifacts.ArtifactVerification performanceVerification) {
        public Result {
            profile = Objects.requireNonNull(profile, "profile must not be null");
            validation = Objects.requireNonNull(validation, "validation must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
            verification = Objects.requireNonNull(verification, "verification must not be null");
            performance = Objects.requireNonNull(performance, "performance must not be null");
            performanceArtifacts = Objects.requireNonNull(performanceArtifacts, "performanceArtifacts must not be null");
            performanceVerification = Objects.requireNonNull(
                    performanceVerification,
                    "performanceVerification must not be null");
        }

        public boolean validationPassed() {
            return validation.passed();
        }

        public boolean performancePassed() {
            return performance.passed();
        }

        public boolean passed() {
            return validationPassed()
                    && verification.passed()
                    && performancePassed()
                    && performanceVerification.passed();
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (passed()) {
                return "Profile `" + profile.id() + "` validation gate passed: " + validation.message();
            }
            if (!verification.passed()) {
                return "Profile `" + profile.id()
                        + "` validation gate failed because artifacts did not verify: "
                        + verification.message();
            }
            if (!performanceVerification.passed()) {
                return "Profile `" + profile.id()
                        + "` validation gate failed because performance artifacts did not verify: "
                        + performanceVerification.message();
            }
            if (!performancePassed()) {
                return "Profile `" + profile.id() + "` validation gate failed: " + performance.message();
            }
            return "Profile `" + profile.id() + "` validation gate failed: " + validation.message();
        }

        public String markdown() {
            return "# Training Report Quality Profile Validation Gate\n\n"
                    + "- Profile: `" + profile.id() + "` (" + profile.displayName() + ")\n"
                    + "- Passed: `" + passed() + "`\n"
                    + "- Validation passed: `" + validationPassed() + "`\n"
                    + "- Artifact verification: `" + verification.passed() + "`\n"
                    + "- Performance passed: `" + performancePassed() + "`\n"
                    + "- Performance artifact verification: `" + performanceVerification.passed() + "`\n"
                    + "- Message: " + message() + "\n\n"
                    + validation.markdown()
                    + "\n"
                    + performance.markdown();
        }

        public String junitXml() {
            return validation.junitXml();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("validationPassed", validationPassed());
            map.put("performancePassed", performancePassed());
            map.put("message", message());
            map.put("profile", profile.toMap());
            map.put("validation", validation.toMap());
            map.put("artifacts", artifacts.toMap());
            map.put("verification", verification.toMap());
            map.put("performance", performance.toMap());
            map.put("performanceArtifacts", performanceArtifacts.toMap());
            map.put("performanceVerification", performanceVerification.toMap());
            return Map.copyOf(map);
        }
    }

    public static Result evaluate(Request request) throws IOException {
        Request resolvedRequest = Objects.requireNonNull(request, "request must not be null");
        TrainingReport report = TrainingReportReader.readReport(resolvedRequest.reportFile());
        TrainingReportValidationPolicy.Result validation =
                report.validate(resolvedRequest.profile().validationPolicy());
        TrainingReportValidationArtifacts.ArtifactBundle artifacts =
                TrainingReportValidationArtifacts.write(
                        resolvedRequest.outputDirectory(),
                        validation,
                        resolvedRequest.artifactOptions());
        TrainingReportValidationArtifacts.ArtifactVerification verification =
                TrainingReportValidationArtifacts.verify(artifacts);
        TrainingReportPerformanceGate.Result performance =
                resolvedRequest.profile().performanceGate(report);
        TrainingReportPerformanceGateArtifacts.ArtifactBundle performanceArtifacts =
                TrainingReportPerformanceGateArtifacts.write(
                        resolvedRequest.outputDirectory().resolve("performance"),
                        performance);
        TrainingReportPerformanceGateArtifacts.ArtifactVerification performanceVerification =
                TrainingReportPerformanceGateArtifacts.verify(performanceArtifacts);
        return new Result(
                resolvedRequest.profile(),
                validation,
                artifacts,
                verification,
                performance,
                performanceArtifacts,
                performanceVerification);
    }

    public static Result evaluate(
            Path reportFile,
            TrainingReportQualityProfile profile,
            Path outputDirectory) throws IOException {
        return evaluate(Request.of(reportFile, profile, outputDirectory));
    }

    public static Result evaluateProfile(
            Path reportFile,
            String profileId,
            Path outputDirectory) throws IOException {
        return evaluate(Request.ofProfile(reportFile, profileId, outputDirectory));
    }

    public static Result evaluateCatalogProfile(
            Path reportFile,
            TrainingReportQualityProfileCatalog catalog,
            String profileId,
            Path outputDirectory) throws IOException {
        return evaluate(Request.ofCatalogProfile(reportFile, catalog, profileId, outputDirectory));
    }

    public static Result evaluateCatalogProfile(
            Path reportFile,
            Path catalogJsonFile,
            String profileId,
            Path outputDirectory) throws IOException {
        return evaluate(Request.ofCatalogProfile(reportFile, catalogJsonFile, profileId, outputDirectory));
    }
}
