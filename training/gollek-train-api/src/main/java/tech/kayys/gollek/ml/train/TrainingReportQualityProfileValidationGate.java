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
    }

    public record Result(
            TrainingReportQualityProfile profile,
            TrainingReportValidationPolicy.Result validation,
            TrainingReportValidationArtifacts.ArtifactBundle artifacts,
            TrainingReportValidationArtifacts.ArtifactVerification verification) {
        public Result {
            profile = Objects.requireNonNull(profile, "profile must not be null");
            validation = Objects.requireNonNull(validation, "validation must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean validationPassed() {
            return validation.passed();
        }

        public boolean passed() {
            return validationPassed() && verification.passed();
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
            return "Profile `" + profile.id() + "` validation gate failed: " + validation.message();
        }

        public String markdown() {
            return "# Training Report Quality Profile Validation Gate\n\n"
                    + "- Profile: `" + profile.id() + "` (" + profile.displayName() + ")\n"
                    + "- Passed: `" + passed() + "`\n"
                    + "- Validation passed: `" + validationPassed() + "`\n"
                    + "- Artifact verification: `" + verification.passed() + "`\n"
                    + "- Message: " + message() + "\n\n"
                    + validation.markdown();
        }

        public String junitXml() {
            return validation.junitXml();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("validationPassed", validationPassed());
            map.put("message", message());
            map.put("profile", profile.toMap());
            map.put("validation", validation.toMap());
            map.put("artifacts", artifacts.toMap());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public static Result evaluate(Request request) throws IOException {
        Request resolvedRequest = Objects.requireNonNull(request, "request must not be null");
        TrainingReportValidationPolicy.Result validation =
                TrainingReportReader.readReport(resolvedRequest.reportFile())
                        .validate(resolvedRequest.profile().validationPolicy());
        TrainingReportValidationArtifacts.ArtifactBundle artifacts =
                TrainingReportValidationArtifacts.write(
                        resolvedRequest.outputDirectory(),
                        validation,
                        resolvedRequest.artifactOptions());
        TrainingReportValidationArtifacts.ArtifactVerification verification =
                TrainingReportValidationArtifacts.verify(artifacts);
        return new Result(resolvedRequest.profile(), validation, artifacts, verification);
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
}
