package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapValues.objectValue;

import java.util.List;
import java.util.Optional;

/**
 * Compares a persisted complete verification report with freshly revalidated artifacts.
 */
final class TrainingReportPromotionGateVerificationReportRevalidation {
    private TrainingReportPromotionGateVerificationReportRevalidation() {
    }

    static void verifyMatches(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection,
            TrainingReportPromotionGateArtifactPackage.PackageVerification revalidated,
            List<String> failures) {
        if (inspection.passed() != revalidated.passed()) {
            failures.add("Verification report pass status is stale for " + inspection.reportFile()
                    + ": report says " + inspection.passed()
                    + " but revalidation says " + revalidated.passed());
        }
        verifyReportBoolean(
                inspection,
                "manifestVerification",
                "manifestSha256Matches",
                revalidated.manifestVerification().manifestSha256Matches(),
                failures);
        verifyReportBoolean(
                inspection,
                "manifestVerification",
                "artifactBytesMatch",
                revalidated.manifestVerification().artifactBytesMatch(),
                failures);
        verifyReportBoolean(
                inspection,
                "manifestVerification",
                "artifactSha256Match",
                revalidated.manifestVerification().artifactSha256Match(),
                failures);
        verifyReportBoolean(
                inspection,
                "sourceSnapshotVerification",
                "passed",
                revalidated.sourceSnapshotVerification().passed(),
                failures);
    }

    private static void verifyReportBoolean(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection,
            String objectKey,
            String booleanKey,
            boolean actual,
            List<String> failures) {
        Optional<Boolean> recorded = objectValue(inspection.report(), objectKey)
                .flatMap(object -> booleanValue(object, booleanKey));
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add("Verification report " + objectKey + "." + booleanKey + " is stale for "
                    + inspection.reportFile() + ": report says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }
}
