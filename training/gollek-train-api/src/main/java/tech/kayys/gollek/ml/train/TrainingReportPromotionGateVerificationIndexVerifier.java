package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexFileReferences.verifyShape;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireString;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.immutableMap;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.objectValue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reader/verifier for the promotion-gate verification index.
 */
final class TrainingReportPromotionGateVerificationIndexVerifier {
    private TrainingReportPromotionGateVerificationIndexVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection read(Path indexFile)
            throws IOException {
        Path resolvedIndexFile = Objects.requireNonNull(indexFile, "indexFile must not be null")
                .toAbsolutePath()
                .normalize();
        String json = Files.readString(resolvedIndexFile, StandardCharsets.UTF_8);
        Object parsed;
        try {
            parsed = TrainerJsonParser.parse(json);
        } catch (IllegalArgumentException error) {
            throw new IOException("Invalid promotion gate package verification index JSON at "
                    + resolvedIndexFile + ": " + error.getMessage(), error);
        }
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Expected JSON object promotion gate package verification index at "
                    + resolvedIndexFile);
        }
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection(
                resolvedIndexFile,
                TrainingReportArtifactFingerprint.of(resolvedIndexFile).sha256(),
                immutableMap(map));
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification verify(
            Path indexFile,
            String expectedIndexSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection inspection = read(indexFile);
        List<String> failures = new ArrayList<>();
        String normalizedExpectedSha256 = expectedIndexSha256 == null || expectedIndexSha256.isBlank()
                ? null
                : expectedIndexSha256.trim();
        boolean indexSha256Matches = normalizedExpectedSha256 == null
                || normalizedExpectedSha256.equalsIgnoreCase(inspection.indexSha256());
        if (!indexSha256Matches) {
            failures.add("Verification index checksum mismatch for " + inspection.indexFile()
                    + ": expected " + normalizedExpectedSha256
                    + " but found " + inspection.indexSha256());
        }
        boolean schemaValid = verifySchema(inspection, failures);
        boolean referencesValid = verifyReferences(inspection, failures);
        return new TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification(
                inspection,
                normalizedExpectedSha256,
                indexSha256Matches,
                schemaValid,
                schemaValid && referencesValid,
                failures);
    }

    private static boolean verifySchema(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> index = inspection.index();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_FORMAT.equals(inspection.format())) {
            failures.add("Verification index format mismatch for " + inspection.indexFile()
                    + ": expected " + TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_FORMAT
                    + " but found " + inspection.format());
        }
        requireString(index, "packageDirectory", "verification index", failures);
        requireString(index, "reportDirectory", "verification index", failures);
        requireString(index, "decisionStatus", "verification index", failures);
        requireObject(index, "manifest", "verification index", failures);
        Map<String, Object> reports = requireObject(index, "reports", "verification index", failures);
        if (reports != null) {
            requireObject(reports, "json", "verification index reports", failures);
            requireObject(reports, "markdown", "verification index reports", failures);
            requireObject(reports, "junitXml", "verification index reports", failures);
            objectValue(reports, "receipt").ifPresent(receipt ->
                    verifyShape("receipt", receipt, "verification index reports", failures));
        }
        Map<String, Object> sourceSnapshots =
                requireObject(index, "sourceReportSnapshots", "verification index", failures);
        if (sourceSnapshots != null) {
            TrainingReportPromotionGateVerificationIndexSourceSnapshots.requireSchema(sourceSnapshots, failures);
        }
        return failures.size() == before;
    }

    private static boolean verifyReferences(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection inspection,
            List<String> failures) throws IOException {
        int before = failures.size();
        Map<String, Object> index = inspection.index();
        Map<String, Object> manifest = objectValue(index, "manifest").orElse(null);
        if (manifest != null) {
            TrainingReportPromotionGateVerificationIndexFileReferences.verify(
                    "manifest", manifest, null, failures);
        }

        Map<String, Object> reports = objectValue(index, "reports").orElse(null);
        if (reports != null) {
            for (String reportName : List.of("json", "markdown", "junitXml", "receipt")) {
                Map<String, Object> report = objectValue(reports, reportName).orElse(null);
                if (report != null) {
                    TrainingReportPromotionGateVerificationIndexFileReferences.verify(
                            "reports." + reportName, report, null, failures);
                }
            }
        }

        Map<String, Object> sourceSnapshots = objectValue(index, "sourceReportSnapshots").orElse(null);
        if (sourceSnapshots != null) {
            TrainingReportPromotionGateVerificationIndexSourceSnapshots.verifyReferences(sourceSnapshots, failures);
        }
        return failures.size() == before;
    }
}
