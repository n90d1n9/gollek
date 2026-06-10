package tech.kayys.gollek.ml.train;

import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageInspection;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.SourceReportSnapshot;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle;

final class TrainingReportPromotionGateArtifactPackageMaps {
    private TrainingReportPromotionGateArtifactPackageMaps() {
    }

    static Map<String, Object> verificationIndex(
            VerifiedPackageBundle verifiedPackage,
            Instant generatedAt) {
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        PackageVerification verification = resolvedVerifiedPackage.verification();
        PackageInspection inspection = verification.inspection();
        SourceSnapshotVerification sourceSnapshots = verification.sourceSnapshotVerification();

        Map<String, Object> reports = new LinkedHashMap<>();
        reports.put("json", file(
                resolvedVerifiedPackage.reports().json().reportFile(),
                resolvedVerifiedPackage.reports().json().reportSha256()));
        reports.put("markdown", file(
                resolvedVerifiedPackage.reports().markdown().markdownFile(),
                resolvedVerifiedPackage.reports().markdown().markdownSha256()));
        reports.put("junitXml", file(
                resolvedVerifiedPackage.reports().junitXml().junitXmlFile(),
                resolvedVerifiedPackage.reports().junitXml().junitXmlSha256()));
        if (resolvedVerifiedPackage.reportBundleReceipt() != null) {
            reports.put("receipt", file(
                    resolvedVerifiedPackage.reportBundleReceipt().receiptFile(),
                    resolvedVerifiedPackage.reportBundleReceipt().receiptSha256()));
        }

        Map<String, Object> sourceSnapshotIndex = new LinkedHashMap<>();
        sourceSnapshotIndex.put("expected", sourceSnapshots.expectedSourceReportArtifactNames().size());
        sourceSnapshotIndex.put("present", sourceSnapshots.presentSourceReportArtifactNames().size());
        sourceSnapshotIndex.put("missing", sourceSnapshots.missingSourceReportArtifactNames());
        sourceSnapshotIndex.put("unexpected", sourceSnapshots.unexpectedSourceReportArtifactNames());
        sourceSnapshotIndex.put("snapshots", sourceSnapshots.snapshots().stream()
                .map(TrainingReportPromotionGateArtifactPackageMaps::sourceSnapshotIndex)
                .toList());

        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("packageDirectory", resolvedVerifiedPackage.directory().toString());
        map.put("reportDirectory", resolvedVerifiedPackage.reports().directory().toString());
        map.put("passed", resolvedVerifiedPackage.passed());
        map.put("promotable", resolvedVerifiedPackage.promotable());
        map.put("decisionStatus", inspection.manifest().decisionStatus());
        inspection.manifest().decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
        map.put("manifest", file(
                inspection.manifest().manifestFile(),
                inspection.manifest().manifestSha256()));
        map.put("reports", reports);
        map.put("sourceReportSnapshots", sourceSnapshotIndex);
        map.put("failures", verification.failures());
        return Map.copyOf(map);
    }

    static Map<String, Object> verificationReport(
            PackageVerification verification,
            Instant generatedAt) {
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.putAll(resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    static Map<String, Object> verificationIndexReceipt(
            VerificationIndexVerification verification,
            Instant generatedAt) {
        VerificationIndexVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_RECEIPT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedVerification.passed());
        map.put("indexFile", resolvedVerification.inspection().indexFile().toString());
        map.put("indexSha256", resolvedVerification.inspection().indexSha256());
        map.put("indexSha256Matches", resolvedVerification.indexSha256Matches());
        map.put("schemaValid", resolvedVerification.schemaValid());
        map.put("referencedSha256Match", resolvedVerification.referencedSha256Match());
        map.put("failures", resolvedVerification.failures());
        map.put("verification", resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    static Map<String, Object> verificationReportBundleReceipt(
            VerificationReportBundleVerification verification,
            Instant generatedAt) {
        VerificationReportBundleVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        VerificationReportBundleInspection inspection = resolvedVerification.inspection();
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportPromotionGateArtifactPackage.VERIFICATION_REPORT_BUNDLE_RECEIPT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedVerification.passed());
        map.put("reportDirectory", inspection.directory().toString());
        map.put("jsonReportFile", inspection.json().reportFile().toString());
        map.put("jsonReportSha256", inspection.json().reportSha256());
        map.put("markdownFile", inspection.markdownFile().toString());
        map.put("markdownSha256", inspection.markdownSha256());
        map.put("junitXmlFile", inspection.junitXmlFile().toString());
        map.put("junitXmlSha256", inspection.junitXmlSha256());
        map.put("jsonReportVerified", resolvedVerification.jsonReportVerified());
        map.put("markdownMatchesRendered", resolvedVerification.markdownMatchesRendered());
        map.put("junitXmlMatchesRendered", resolvedVerification.junitXmlMatchesRendered());
        map.put("failures", resolvedVerification.failures());
        map.put("verification", resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    static Map<String, Object> verificationEvidenceReceipt(
            VerificationEvidenceVerification verification,
            Instant generatedAt) {
        VerificationEvidenceVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_RECEIPT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedVerification.passed());
        map.put("evidenceFile", resolvedVerification.inspection().evidenceFile().toString());
        map.put("evidenceSha256", resolvedVerification.inspection().evidenceSha256());
        map.put("evidenceSha256Matches", resolvedVerification.evidenceSha256Matches());
        map.put("schemaValid", resolvedVerification.schemaValid());
        map.put("evidenceFilesSha256Match", resolvedVerification.evidenceFilesSha256Match());
        map.put("packageArtifactsSha256Match", resolvedVerification.packageArtifactsSha256Match());
        map.put("failures", resolvedVerification.failures());
        map.put("verification", resolvedVerification.toMap());
        return Map.copyOf(map);
    }

    static Map<String, Object> verificationIndexPackageAuditReport(
            VerificationIndexPackageAudit audit,
            Instant generatedAt) {
        VerificationIndexPackageAudit resolvedAudit = Objects.requireNonNull(audit, "audit must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportPromotionGateArtifactPackage.VERIFICATION_INDEX_PACKAGE_AUDIT_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("passed", resolvedAudit.passed());
        map.put("indexFile", resolvedAudit.indexVerification().inspection().indexFile().toString());
        map.put("indexSha256", resolvedAudit.indexVerification().inspection().indexSha256());
        map.put("indexPassed", resolvedAudit.indexVerification().passed());
        map.put("packagePassed", resolvedAudit.packageVerification() != null
                && resolvedAudit.packageVerification().passed());
        map.put("failures", resolvedAudit.failures());
        map.put("audit", resolvedAudit.toMap());
        return Map.copyOf(map);
    }

    static Map<String, Object> verificationEvidenceManifest(
            VerifiedPackageBundle verifiedPackage,
            Instant generatedAt) {
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        PackageInspection inspection = resolvedVerifiedPackage.verification().inspection();

        Map<String, Object> evidenceFiles = new LinkedHashMap<>();
        evidenceFiles.put("manifest", file(
                inspection.manifest().manifestFile(),
                inspection.manifest().manifestSha256()));
        evidenceFiles.put("verificationJson", file(
                resolvedVerifiedPackage.reports().json().reportFile(),
                resolvedVerifiedPackage.reports().json().reportSha256()));
        evidenceFiles.put("verificationMarkdown", file(
                resolvedVerifiedPackage.reports().markdown().markdownFile(),
                resolvedVerifiedPackage.reports().markdown().markdownSha256()));
        evidenceFiles.put("verificationJunitXml", file(
                resolvedVerifiedPackage.reports().junitXml().junitXmlFile(),
                resolvedVerifiedPackage.reports().junitXml().junitXmlSha256()));
        if (resolvedVerifiedPackage.reportBundleReceipt() != null) {
            evidenceFiles.put("verificationReportBundleReceipt", file(
                    resolvedVerifiedPackage.reportBundleReceipt().receiptFile(),
                    resolvedVerifiedPackage.reportBundleReceipt().receiptSha256()));
        }
        if (resolvedVerifiedPackage.index() != null) {
            evidenceFiles.put("verificationIndex", file(
                    resolvedVerifiedPackage.index().indexFile(),
                    resolvedVerifiedPackage.index().indexSha256()));
        }
        if (resolvedVerifiedPackage.receipt() != null) {
            evidenceFiles.put("verificationIndexReceipt", file(
                    resolvedVerifiedPackage.receipt().receiptFile(),
                    resolvedVerifiedPackage.receipt().receiptSha256()));
        }
        if (resolvedVerifiedPackage.packageAuditReport() != null) {
            evidenceFiles.put("verificationIndexPackageAudit", file(
                    resolvedVerifiedPackage.packageAuditReport().reportFile(),
                    resolvedVerifiedPackage.packageAuditReport().reportSha256()));
        }

        Map<String, Object> map = new LinkedHashMap<>();
        map.put("format", TrainingReportPromotionGateArtifactPackage.VERIFICATION_EVIDENCE_FORMAT);
        map.put("generatedAt", resolvedGeneratedAt.toString());
        map.put("packageDirectory", resolvedVerifiedPackage.directory().toString());
        map.put("reportDirectory", resolvedVerifiedPackage.reports().directory().toString());
        map.put("passed", resolvedVerifiedPackage.passed());
        map.put("promotable", resolvedVerifiedPackage.promotable());
        map.put("decisionStatus", inspection.manifest().decisionStatus());
        inspection.manifest().decisionCandidate().ifPresent(candidate -> map.put("decisionCandidate", candidate));
        map.put("evidenceFiles", evidenceFiles);
        map.put("packageArtifacts", packageArtifacts(inspection.manifest().artifacts()));
        map.put("sourceReportSnapshots", resolvedVerifiedPackage.verification()
                .sourceSnapshotVerification()
                .toMap());
        return Map.copyOf(map);
    }

    private static Map<String, Object> packageArtifacts(
            Map<String, TrainingReportPromotionGateArtifactManifest.ArtifactEntry> artifacts) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<String, TrainingReportPromotionGateArtifactManifest.ArtifactEntry> entry : artifacts.entrySet()) {
            values.put(entry.getKey(), entry.getValue().toMap());
        }
        return Map.copyOf(values);
    }

    private static Map<String, Object> file(Path file, String sha256) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("file", Objects.requireNonNull(file, "file must not be null")
                .toAbsolutePath()
                .normalize()
                .toString());
        map.put("sha256", Objects.requireNonNull(sha256, "sha256 must not be null"));
        return Map.copyOf(map);
    }

    private static Map<String, Object> sourceSnapshotIndex(SourceReportSnapshot snapshot) {
        SourceReportSnapshot resolvedSnapshot = Objects.requireNonNull(snapshot, "snapshot must not be null");
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("role", resolvedSnapshot.role());
        map.put("name", resolvedSnapshot.name());
        map.put("artifact", resolvedSnapshot.artifact().name());
        map.put("file", resolvedSnapshot.snapshotFile().toString());
        map.put("bytes", resolvedSnapshot.artifact().bytes());
        map.put("sha256", resolvedSnapshot.artifact().sha256());
        map.put("manifestBytesMatchSource", resolvedSnapshot.manifestBytesMatchSource());
        map.put("manifestSha256MatchesSource", resolvedSnapshot.manifestSha256MatchesSource());
        return Map.copyOf(map);
    }
}
