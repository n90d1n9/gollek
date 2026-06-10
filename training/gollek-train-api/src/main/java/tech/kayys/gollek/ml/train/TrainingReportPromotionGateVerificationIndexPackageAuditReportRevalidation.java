package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexValues.booleanValue;

import java.util.List;
import java.util.Optional;

/**
 * Compares persisted verification-index package audit reports with fresh audit results.
 */
final class TrainingReportPromotionGateVerificationIndexPackageAuditReportRevalidation {
    private TrainingReportPromotionGateVerificationIndexPackageAuditReportRevalidation() {
    }

    static void verifyMatches(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection,
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit revalidated,
            List<String> failures) {
        if (inspection.passed() != revalidated.passed()) {
            failures.add("Verification index package audit report pass status is stale for "
                    + inspection.reportFile() + ": report says " + inspection.passed()
                    + " but revalidation says " + revalidated.passed());
        }
        verifyReportBoolean(inspection, "indexPassed", revalidated.indexVerification().passed(), failures);
        boolean packagePassed = revalidated.packageVerification() != null
                && revalidated.packageVerification().passed();
        verifyReportBoolean(inspection, "packagePassed", packagePassed, failures);
    }

    private static void verifyReportBoolean(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection,
            String key,
            boolean actual,
            List<String> failures) {
        Optional<Boolean> recorded = booleanValue(inspection.report(), key);
        if (recorded.isPresent() && recorded.get().booleanValue() != actual) {
            failures.add("Verification index package audit report " + key + " is stale for "
                    + inspection.reportFile() + ": report says " + recorded.get()
                    + " but revalidation says " + actual);
        }
    }
}
