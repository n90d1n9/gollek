package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.immutableMap;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reader/verifier for persisted verification-index package-audit reports.
 */
final class TrainingReportPromotionGateVerificationIndexPackageAuditReportVerifier {
    private TrainingReportPromotionGateVerificationIndexPackageAuditReportVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection read(
            Path reportFile) throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedReportFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate verification index package audit report JSON at "
                    + resolvedReportFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate verification index package audit report at "
                    + resolvedReportFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection(
                resolvedReportFile,
                TrainingReportArtifactFingerprint.of(resolvedReportFile).sha256(),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification verify(
            Path reportFile,
            String expectedReportSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection =
                read(reportFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedReportSha256 == null || expectedReportSha256.isBlank()
                ? null
                : expectedReportSha256.trim();
        boolean reportSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.reportSha256());
        if (!reportSha256Matches) {
            failures.add("Verification index package audit report checksum mismatch for "
                    + inspection.reportFile() + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.reportSha256());
        }
        boolean schemaValid =
                TrainingReportPromotionGateVerificationIndexPackageAuditReportSchema.verify(inspection, failures);
        TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit audit = null;
        if (schemaValid && inspection.indexFile() != null) {
            try {
                audit = TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackage(
                        inspection.indexFile(),
                        inspection.indexSha256());
                failures.addAll(audit.failures());
                TrainingReportPromotionGateVerificationIndexPackageAuditReportRevalidation.verifyMatches(
                        inspection,
                        audit,
                        failures);
            } catch (IOException error) {
                failures.add("Verification index package audit report could not revalidate index "
                        + inspection.indexFile() + ": " + error.getMessage());
            }
        }
        boolean auditRevalidated = audit != null && audit.passed();
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification(
                inspection,
                normalizedExpectedSha256,
                reportSha256Matches,
                schemaValid,
                auditRevalidated,
                audit,
                failures);
    }
}
