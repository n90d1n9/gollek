package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireIterable;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateVerificationIndexRequirements.requireString;

import java.util.List;
import java.util.Map;

/**
 * Schema contract for persisted verification-index package audit reports.
 */
final class TrainingReportPromotionGateVerificationIndexPackageAuditReportSchema {
    private TrainingReportPromotionGateVerificationIndexPackageAuditReportSchema() {
    }

    static boolean verify(
            TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> report = inspection.report();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT
                .equals(inspection.format())) {
            failures.add("Verification index package audit report format mismatch for " + inspection.reportFile()
                    + ": expected "
                    + TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT
                    + " but found " + inspection.format());
        }
        String owner = "verification index package audit report";
        requireString(report, "indexFile", owner, failures);
        requireString(report, "indexSha256", owner, failures);
        requireBoolean(report, "passed", owner, failures);
        requireBoolean(report, "indexPassed", owner, failures);
        requireBoolean(report, "packagePassed", owner, failures);
        requireIterable(report, "failures", owner, failures);
        requireObject(report, "audit", owner, failures);
        return failures.size() == before;
    }
}
