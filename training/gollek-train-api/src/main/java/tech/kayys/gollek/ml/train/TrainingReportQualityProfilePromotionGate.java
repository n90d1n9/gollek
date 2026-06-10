package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Profile-aware wrapper around the promotion gate for CI/CD quality contracts.
 */
public final class TrainingReportQualityProfilePromotionGate {
    private TrainingReportQualityProfilePromotionGate() {
    }

    public record Request(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportQualityProfile profile,
            Path outputDirectory,
            TrainingReportPromotionArtifacts.Options artifactOptions) {
        public Request {
            profile = profile == null ? TrainingReportQualityProfile.productionPromotion() : profile;
            artifactOptions = artifactOptions == null
                    ? TrainingReportPromotionArtifacts.Options.defaults()
                    : artifactOptions;
        }

        public static Request of(
                Map<String, Path> reportFiles,
                String baselineName,
                TrainingReportQualityProfile profile,
                Path outputDirectory) {
            return new Request(
                    reportFiles,
                    baselineName,
                    profile,
                    outputDirectory,
                    TrainingReportPromotionArtifacts.Options.defaults());
        }

        public static Request ofProfile(
                Map<String, Path> reportFiles,
                String baselineName,
                String profileId,
                Path outputDirectory) {
            return of(
                    reportFiles,
                    baselineName,
                    TrainingReportQualityProfile.require(profileId),
                    outputDirectory);
        }

        TrainingReportPromotionGate.Request toPromotionGateRequest() {
            return new TrainingReportPromotionGate.Request(
                    reportFiles,
                    baselineName,
                    profile.promotionPolicy(),
                    outputDirectory,
                    artifactOptions);
        }
    }

    public record Result(
            TrainingReportQualityProfile profile,
            TrainingReportPromotionGate.Result gate) {
        public Result {
            profile = Objects.requireNonNull(profile, "profile must not be null");
            gate = Objects.requireNonNull(gate, "gate must not be null");
        }

        public TrainingReportPortfolio.PromotionReview review() {
            return gate.review();
        }

        public TrainingReportPortfolio.PromotionDecision decision() {
            return gate.decision();
        }

        public TrainingReportPromotionArtifacts.ArtifactBundle artifacts() {
            return gate.artifacts();
        }

        public TrainingReportPromotionArtifacts.ArtifactVerification verification() {
            return gate.verification();
        }

        public TrainingReportPromotionArtifacts.SourceVerification sourceVerification() {
            return gate.sourceVerification();
        }

        public boolean promotable() {
            return gate.promotable();
        }

        public boolean passed() {
            return gate.passed();
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            return "Profile `" + profile.id() + "`: " + gate.message();
        }

        public String markdown() {
            return "# Training Report Quality Profile Gate\n\n"
                    + "- Profile: `" + profile.id() + "` (" + profile.displayName() + ")\n"
                    + "- Passed: `" + passed() + "`\n"
                    + "- Promotable: `" + promotable() + "`\n"
                    + "- Message: " + message() + "\n\n"
                    + gate.markdown();
        }

        public String junitXml() {
            return gate.junitXml();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("promotable", promotable());
            map.put("message", message());
            map.put("profile", profile.toMap());
            map.put("gate", gate.toMap());
            map.put("review", review().toMap());
            map.put("decision", decision().toMap());
            map.put("artifacts", artifacts().toMap());
            map.put("verification", verification().toMap());
            map.put("sourceVerification", sourceVerification().toMap());
            return Map.copyOf(map);
        }
    }

    public static Result evaluate(Request request) throws IOException {
        Request resolvedRequest = Objects.requireNonNull(request, "request must not be null");
        TrainingReportPromotionGate.Result gate =
                TrainingReportPromotionGate.evaluate(resolvedRequest.toPromotionGateRequest());
        return new Result(resolvedRequest.profile(), gate);
    }

    public static Result evaluate(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportQualityProfile profile,
            Path outputDirectory) throws IOException {
        return evaluate(Request.of(reportFiles, baselineName, profile, outputDirectory));
    }

    public static Result evaluateProfile(
            Map<String, Path> reportFiles,
            String baselineName,
            String profileId,
            Path outputDirectory) throws IOException {
        return evaluate(Request.ofProfile(reportFiles, baselineName, profileId, outputDirectory));
    }
}
