package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapRequirements.requireBoolean;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapRequirements.requireIterable;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapRequirements.requireObject;
import static tech.kayys.gollek.ml.train.TrainingReportPromotionGateMapRequirements.requireString;

import java.util.List;
import java.util.Map;

/**
 * Schema contract for persisted complete package verification reports.
 */
final class TrainingReportPromotionGateVerificationReportSchema {
    private TrainingReportPromotionGateVerificationReportSchema() {
    }

    static boolean verify(
            TrainingReportPromotionGateArtifactPackage.VerificationReportInspection inspection,
            List<String> failures) {
        int before = failures.size();
        Map<String, Object> report = inspection.report();
        if (!TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_FORMAT.equals(inspection.format())) {
            failures.add("Verification report format mismatch for " + inspection.reportFile()
                    + ": expected " + TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_FORMAT
                    + " but found " + inspection.format());
        }
        requireBoolean(report, "passed", "verification report", failures);
        requireIterable(report, "failures", "verification report", failures);
        verifyManifestVerification(report, failures);
        verifySourceSnapshotVerification(report, failures);
        verifyPackageInspection(report, failures);
        return failures.size() == before;
    }

    private static void verifyManifestVerification(Map<String, Object> report, List<String> failures) {
        Map<String, Object> manifestVerification =
                requireObject(report, "manifestVerification", "verification report", failures);
        if (manifestVerification == null) {
            return;
        }
        String owner = "verification report manifestVerification";
        requireString(manifestVerification, "actualManifestSha256", owner, failures);
        requireBoolean(manifestVerification, "manifestSha256Matches", owner, failures);
        requireBoolean(manifestVerification, "artifactBytesMatch", owner, failures);
        requireBoolean(manifestVerification, "artifactSha256Match", owner, failures);
        Map<String, Object> manifestInspection =
                requireObject(manifestVerification, "inspection", owner, failures);
        if (manifestInspection != null) {
            requireString(manifestInspection, "directory", "verification report manifest inspection", failures);
            requireString(manifestInspection, "manifestFile", "verification report manifest inspection", failures);
        }
    }

    private static void verifySourceSnapshotVerification(Map<String, Object> report, List<String> failures) {
        Map<String, Object> sourceSnapshotVerification =
                requireObject(report, "sourceSnapshotVerification", "verification report", failures);
        if (sourceSnapshotVerification == null) {
            return;
        }
        String owner = "verification report sourceSnapshotVerification";
        requireBoolean(sourceSnapshotVerification, "passed", owner, failures);
        requireIterable(sourceSnapshotVerification, "snapshots", owner, failures);
    }

    private static void verifyPackageInspection(Map<String, Object> report, List<String> failures) {
        Map<String, Object> packageInspection = requireObject(report, "inspection", "verification report", failures);
        if (packageInspection != null) {
            requireString(packageInspection, "directory", "verification report inspection", failures);
        }
    }
}
