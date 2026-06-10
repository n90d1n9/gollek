package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * One-call CI gate that validates all reports and then evaluates promotion with one quality profile.
 */
public final class TrainingReportQualityProfileCiGate {
    private TrainingReportQualityProfileCiGate() {
    }

    public record ArtifactOptions(
            TrainingReportValidationArtifacts.Options validation,
            TrainingReportQualityProfileValidationGateArtifacts.Options profileValidation,
            TrainingReportPromotionArtifacts.Options promotion,
            TrainingReportQualityProfilePromotionGateArtifacts.Options profilePromotion) {
        public ArtifactOptions {
            validation = validation == null ? TrainingReportValidationArtifacts.Options.defaults() : validation;
            profileValidation = profileValidation == null
                    ? TrainingReportQualityProfileValidationGateArtifacts.Options.defaults()
                    : profileValidation;
            promotion = promotion == null ? TrainingReportPromotionArtifacts.Options.defaults() : promotion;
            profilePromotion = profilePromotion == null
                    ? TrainingReportQualityProfilePromotionGateArtifacts.Options.defaults()
                    : profilePromotion;
        }

        public static ArtifactOptions defaults() {
            return new ArtifactOptions(
                    TrainingReportValidationArtifacts.Options.defaults(),
                    TrainingReportQualityProfileValidationGateArtifacts.Options.defaults(),
                    TrainingReportPromotionArtifacts.Options.defaults(),
                    TrainingReportQualityProfilePromotionGateArtifacts.Options.defaults());
        }
    }

    public record Request(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportQualityProfile profile,
            Path outputDirectory,
            ArtifactOptions artifactOptions) {
        public Request {
            reportFiles = normalizeReportFiles(reportFiles);
            if (baselineName == null || baselineName.isBlank()) {
                throw new IllegalArgumentException("baselineName must not be blank");
            }
            baselineName = baselineName.trim();
            if (!reportFiles.containsKey(baselineName)) {
                throw new IllegalArgumentException("baselineName must match one report name: " + baselineName);
            }
            profile = profile == null ? TrainingReportQualityProfile.productionPromotion() : profile;
            outputDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                    .toAbsolutePath()
                    .normalize();
            artifactOptions = artifactOptions == null ? ArtifactOptions.defaults() : artifactOptions;
        }

        public static Request of(
                Map<String, Path> reportFiles,
                String baselineName,
                TrainingReportQualityProfile profile,
                Path outputDirectory) {
            return new Request(reportFiles, baselineName, profile, outputDirectory, ArtifactOptions.defaults());
        }

        public static Request ofProfile(
                Map<String, Path> reportFiles,
                String baselineName,
                String profileId,
                Path outputDirectory) {
            return of(reportFiles, baselineName, TrainingReportQualityProfile.require(profileId), outputDirectory);
        }
    }

    public record Result(
            TrainingReportQualityProfile profile,
            Map<String, TrainingReportQualityProfileValidationGate.Result> validations,
            Map<String, TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle> validationArtifacts,
            Map<String, TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification> validationArtifactVerifications,
            TrainingReportQualityProfilePromotionGate.Result promotion,
            TrainingReportQualityProfilePromotionGateArtifacts.ArtifactBundle promotionArtifacts,
            TrainingReportQualityProfilePromotionGateArtifacts.ArtifactVerification promotionArtifactVerification) {
        public Result {
            profile = Objects.requireNonNull(profile, "profile must not be null");
            validations = Map.copyOf(Objects.requireNonNull(validations, "validations must not be null"));
            validationArtifacts = Map.copyOf(Objects.requireNonNull(
                    validationArtifacts,
                    "validationArtifacts must not be null"));
            validationArtifactVerifications = Map.copyOf(Objects.requireNonNull(
                    validationArtifactVerifications,
                    "validationArtifactVerifications must not be null"));
            promotion = Objects.requireNonNull(promotion, "promotion must not be null");
            promotionArtifacts = Objects.requireNonNull(promotionArtifacts, "promotionArtifacts must not be null");
            promotionArtifactVerification = Objects.requireNonNull(
                    promotionArtifactVerification,
                    "promotionArtifactVerification must not be null");
        }

        public boolean validationPassed() {
            return validations.values().stream().allMatch(TrainingReportQualityProfileValidationGate.Result::passed);
        }

        public boolean promotionPassed() {
            return promotion.passed();
        }

        public boolean artifactsVerified() {
            return promotionArtifactVerification.passed()
                    && validationArtifactVerifications.values().stream()
                            .allMatch(TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification::passed);
        }

        public boolean passed() {
            return validationPassed() && promotionPassed() && artifactsVerified();
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public List<String> failedValidationNames() {
            return validations.entrySet().stream()
                    .filter(entry -> !entry.getValue().passed())
                    .map(Map.Entry::getKey)
                    .toList();
        }

        public String message() {
            if (passed()) {
                return "Quality profile CI gate passed for `" + profile.id() + "`.";
            }
            StringBuilder message = new StringBuilder("Quality profile CI gate failed for `")
                    .append(profile.id())
                    .append("`.");
            List<String> failedValidations = failedValidationNames();
            if (!failedValidations.isEmpty()) {
                message.append(" Failed validations: ").append(failedValidations).append('.');
            }
            if (!promotionPassed()) {
                message.append(" Promotion failed: ").append(promotion.message());
            }
            if (!artifactsVerified()) {
                message.append(" Artifact verification failed.");
            }
            return message.toString();
        }

        public String markdown() {
            StringBuilder markdown = new StringBuilder();
            appendLine(markdown, "# Gollek Training Quality Profile CI Gate");
            appendLine(markdown, "");
            appendLine(markdown, "**Profile:** `" + profile.id() + "` (" + profile.displayName() + ")");
            appendLine(markdown, "**Gate:** `" + (passed() ? "PASS" : "FAIL") + "`");
            appendLine(markdown, "**Validation:** `" + (validationPassed() ? "PASS" : "FAIL") + "`");
            appendLine(markdown, "**Promotion:** `" + (promotionPassed() ? "PASS" : "FAIL") + "`");
            appendLine(markdown, "**Artifacts verified:** `" + (artifactsVerified() ? "PASS" : "FAIL") + "`");
            appendLine(markdown, "");
            appendLine(markdown, "## Validations");
            appendLine(markdown, "");
            appendLine(markdown, "| Report | Passed | Artifact Verified | Failures |");
            appendLine(markdown, "| --- | --- | --- | --- |");
            for (Map.Entry<String, TrainingReportQualityProfileValidationGate.Result> entry : validations.entrySet()) {
                TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification verification =
                        validationArtifactVerifications.get(entry.getKey());
                appendLine(markdown, validationRow(entry.getKey(), entry.getValue(), verification));
            }
            appendLine(markdown, "");
            appendLine(markdown, "## Promotion");
            appendLine(markdown, "");
            appendLine(markdown, "- Passed: `" + promotionPassed() + "`");
            appendLine(markdown, "- Promotable: `" + promotion.promotable() + "`");
            appendLine(markdown, "- Candidate: `"
                    + promotion.decision().candidate().map(TrainingReportPortfolio.Entry::name).orElse("none")
                    + "`");
            appendLine(markdown, "- Profile promotion artifact verified: `"
                    + promotionArtifactVerification.passed() + "`");
            appendLine(markdown, "");
            appendLine(markdown, message());
            return markdown.toString();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> validationMap = new LinkedHashMap<>();
            for (Map.Entry<String, TrainingReportQualityProfileValidationGate.Result> entry : validations.entrySet()) {
                validationMap.put(entry.getKey(), entry.getValue().toMap());
            }
            Map<String, Object> validationArtifactMap = new LinkedHashMap<>();
            for (Map.Entry<String, TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle> entry
                    : validationArtifacts.entrySet()) {
                validationArtifactMap.put(entry.getKey(), entry.getValue().toMap());
            }
            Map<String, Object> validationVerificationMap = new LinkedHashMap<>();
            for (Map.Entry<String, TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification> entry
                    : validationArtifactVerifications.entrySet()) {
                validationVerificationMap.put(entry.getKey(), entry.getValue().toMap());
            }

            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("validationPassed", validationPassed());
            map.put("promotionPassed", promotionPassed());
            map.put("artifactsVerified", artifactsVerified());
            map.put("message", message());
            map.put("profile", profile.toMap());
            map.put("failedValidationNames", failedValidationNames());
            map.put("validations", validationMap);
            map.put("validationArtifacts", validationArtifactMap);
            map.put("validationArtifactVerifications", validationVerificationMap);
            map.put("promotion", promotion.toMap());
            map.put("promotionArtifacts", promotionArtifacts.toMap());
            map.put("promotionArtifactVerification", promotionArtifactVerification.toMap());
            return Map.copyOf(map);
        }
    }

    public static Result evaluate(Request request) throws IOException {
        Request resolvedRequest = Objects.requireNonNull(request, "request must not be null");
        Map<String, TrainingReportQualityProfileValidationGate.Result> validations = new LinkedHashMap<>();
        Map<String, TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle> validationArtifacts =
                new LinkedHashMap<>();
        Map<String, TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification> validationVerifications =
                new LinkedHashMap<>();

        int index = 0;
        for (Map.Entry<String, Path> entry : resolvedRequest.reportFiles().entrySet()) {
            Path validationDirectory = resolvedRequest.outputDirectory()
                    .resolve("validations")
                    .resolve(safeDirectoryName(entry.getKey(), index++));
            TrainingReportQualityProfileValidationGate.Result validation =
                    TrainingReportQualityProfileValidationGate.evaluate(new TrainingReportQualityProfileValidationGate.Request(
                            entry.getValue(),
                            resolvedRequest.profile(),
                            validationDirectory,
                            resolvedRequest.artifactOptions().validation()));
            TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle profileArtifacts =
                    TrainingReportQualityProfileValidationGateArtifacts.write(
                            validationDirectory,
                            validation,
                            resolvedRequest.artifactOptions().profileValidation());
            validations.put(entry.getKey(), validation);
            validationArtifacts.put(entry.getKey(), profileArtifacts);
            validationVerifications.put(
                    entry.getKey(),
                    TrainingReportQualityProfileValidationGateArtifacts.verify(profileArtifacts));
        }

        Path promotionDirectory = resolvedRequest.outputDirectory().resolve("promotion");
        TrainingReportQualityProfilePromotionGate.Result promotion =
                TrainingReportQualityProfilePromotionGate.evaluate(new TrainingReportQualityProfilePromotionGate.Request(
                        resolvedRequest.reportFiles(),
                        resolvedRequest.baselineName(),
                        resolvedRequest.profile(),
                        promotionDirectory,
                        resolvedRequest.artifactOptions().promotion()));
        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactBundle promotionArtifacts =
                TrainingReportQualityProfilePromotionGateArtifacts.write(
                        promotionDirectory,
                        promotion,
                        resolvedRequest.artifactOptions().profilePromotion());
        TrainingReportQualityProfilePromotionGateArtifacts.ArtifactVerification promotionVerification =
                TrainingReportQualityProfilePromotionGateArtifacts.verify(promotionArtifacts);

        return new Result(
                resolvedRequest.profile(),
                validations,
                validationArtifacts,
                validationVerifications,
                promotion,
                promotionArtifacts,
                promotionVerification);
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
            normalized.put(
                    name.trim(),
                    Objects.requireNonNull(entry.getValue(), "report file path must not be null")
                            .toAbsolutePath()
                            .normalize());
        }
        return Map.copyOf(normalized);
    }

    private static String safeDirectoryName(String value, int index) {
        String slug = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        slug = slug.replaceAll("[^a-z0-9._-]+", "-").replaceAll("-+", "-");
        while (slug.startsWith("-") || slug.startsWith(".")) {
            slug = slug.substring(1);
        }
        while (slug.endsWith("-") || slug.endsWith(".")) {
            slug = slug.substring(0, slug.length() - 1);
        }
        if (slug.isBlank()) {
            slug = "report";
        }
        return "%02d-%s".formatted(index + 1, slug);
    }

    private static String validationRow(
            String name,
            TrainingReportQualityProfileValidationGate.Result validation,
            TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification verification) {
        return "| `" + escapeTable(name) + "`"
                + " | `" + validation.passed() + "`"
                + " | `" + (verification != null && verification.passed()) + "`"
                + " | `" + escapeTable(String.join(", ", validation.validation().failureCodes())) + "` |";
    }

    private static String escapeTable(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        return value.trim().replace("`", "\\`").replace("|", "\\|").replace("\n", " ");
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
