package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapValues.immutableMap;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reader/verifier for persisted complete package verification JSON reports.
 */
final class TrainingReportPromotionGateVerificationReportVerifier {
    private TrainingReportPromotionGateVerificationReportVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportInspection read(Path reportFile)
            throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReportFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification report JSON at "
                    + resolvedReportFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification report at "
                    + resolvedReportFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportInspection(
                resolvedReportFile,
                TrainingReportArtifactFingerprint.of(resolvedReportFile).sha256(),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportVerification verify(
            Path reportFile,
            String expectedReportSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection = read(reportFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = normalizeExpectedSha256(expectedReportSha256);
        boolean reportSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.reportSha256());
        if (!reportSha256Matches) {
            failures.add("Verification report checksum mismatch for " + inspection.reportFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.reportSha256());
        }
        boolean schemaValid = TrainingReportPromotionGateVerificationReportSchema.verify(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.PackageVerification packageVerification = null;
        if (schemaValid && inspection.packageDirectory() != null) {
            try {
                packageVerification = TrainingReportPromotionGateArtifactPackage.verifyComplete(
                        inspection.packageDirectory(),
                        inspection.manifestSha256(),
                        optionsFromInspection(inspection));
                failures.addAll(packageVerification.failures());
                TrainingReportPromotionGateVerificationReportRevalidation.verifyMatches(
                        inspection,
                        packageVerification,
                        failures);
            } catch (IOException error) {
                failures.add("Verification report could not revalidate package "
                        + inspection.packageDirectory() + ": " + error.getMessage());
            }
        }
        boolean packageRevalidated = packageVerification != null && packageVerification.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportVerification(
                inspection,
                normalizedExpectedSha256,
                reportSha256Matches,
                schemaValid,
                packageRevalidated,
                packageVerification,
                failures);
    }

    private static String normalizeExpectedSha256(String expectedReportSha256) {
        return expectedReportSha256 == null || expectedReportSha256.isBlank()
                ? null
                : expectedReportSha256.trim().toLowerCase(java.util.Locale.ROOT);
    }

    private static TrainingReportPromotionGateArtifactPackage.Options optionsFromInspection(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection) {
        Path manifestFile = inspection.manifestFile();
        if (manifestFile == null || manifestFile.getFileName() == null) {
            return TrainingReportPromotionGateArtifactPackage.Options.defaults();
        }
        return new TrainingReportPromotionGateArtifactPackage.Options(
                TrainingReportPromotionArtifacts.Options.defaults(),
                TrainingReportPromotionGateArtifacts.Options.defaults(),
                new TrainingReportPromotionGateArtifactManifest.Options(manifestFile.getFileName().toString()));
    }
}
