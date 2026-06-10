package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationEvidenceManifest;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceipt;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndex;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReport;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexReceipt;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReport;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundle;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceipt;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle;

final class TrainingReportPromotionGatePackageVerificationArtifacts {
    private TrainingReportPromotionGatePackageVerificationArtifacts() {
    }

    static VerificationReport writeVerificationReport(
            Path reportFile,
            PackageVerification verification) throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReportFile,
                TrainerJson.toJson(TrainingReportPromotionGateArtifactPackageMaps.verificationReport(
                        resolvedVerification,
                        Instant.now())) + "\n");
        return new VerificationReport(
                resolvedReportFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedReportFile),
                resolvedVerification);
    }

    static VerificationMarkdownReport writeVerificationMarkdownReport(
            Path markdownFile,
            PackageVerification verification) throws IOException {
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedMarkdownFile,
                TrainingReportPromotionGatePackageMarkdown.render(resolvedVerification));
        return new VerificationMarkdownReport(
                resolvedMarkdownFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedMarkdownFile),
                resolvedVerification);
    }

    static VerificationJUnitXmlReport writeVerificationJUnitXmlReport(
            Path junitXmlFile,
            PackageVerification verification) throws IOException {
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedJunitXmlFile,
                TrainingReportPromotionGatePackageJUnitXml.render(resolvedVerification));
        return new VerificationJUnitXmlReport(
                resolvedJunitXmlFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedJunitXmlFile),
                resolvedVerification);
    }

    static VerificationReportBundle writeVerificationReports(
            Path reportDirectory,
            PackageVerification verification) throws IOException {
        Path resolvedReportDirectory = Objects.requireNonNull(reportDirectory, "reportDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        PackageVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        return new VerificationReportBundle(
                resolvedReportDirectory,
                writeVerificationReport(
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationReportFile(
                                resolvedReportDirectory),
                        resolvedVerification),
                writeVerificationMarkdownReport(
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationMarkdownFile(
                                resolvedReportDirectory),
                        resolvedVerification),
                writeVerificationJUnitXmlReport(
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationJunitXmlFile(
                                resolvedReportDirectory),
                        resolvedVerification));
    }

    static VerificationReportBundleReceipt writeVerificationReportBundleReceipt(
            Path receiptFile,
            VerificationReportBundleVerification verification) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationReportBundleVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReceiptFile,
                TrainerJson.toJson(TrainingReportPromotionGateArtifactPackageMaps.verificationReportBundleReceipt(
                        resolvedVerification,
                        Instant.now())) + "\n");
        return new VerificationReportBundleReceipt(
                resolvedReceiptFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedReceiptFile),
                resolvedVerification);
    }

    static VerificationReportBundleReceipt verifyVerificationReportBundleAndWriteReceipt(
            Path reportDirectory,
            String expectedJsonReportSha256,
            Path receiptFile) throws IOException {
        VerificationReportBundleVerification verification =
                TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundle(
                        reportDirectory,
                        expectedJsonReportSha256);
        return writeVerificationReportBundleReceipt(receiptFile, verification);
    }

    static VerificationIndex writeVerificationIndex(
            Path indexFile,
            VerifiedPackageBundle verifiedPackage) throws IOException {
        Path resolvedIndexFile = Objects.requireNonNull(indexFile, "indexFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedIndexFile,
                TrainerJson.toJson(TrainingReportPromotionGateArtifactPackageMaps.verificationIndex(
                        resolvedVerifiedPackage,
                        Instant.now())) + "\n");
        return new VerificationIndex(
                resolvedIndexFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedIndexFile));
    }

    static VerificationIndexPackageAudit auditVerificationIndexPackage(
            VerificationIndexVerification indexVerification) throws IOException {
        VerificationIndexVerification resolvedIndexVerification = Objects.requireNonNull(
                indexVerification,
                "indexVerification must not be null");
        List<String> failures = new ArrayList<>(resolvedIndexVerification.failures());
        PackageVerification packageVerification = null;
        Path packageDirectory = resolvedIndexVerification.inspection().packageDirectory();
        if (packageDirectory == null) {
            failures.add("Verification index package audit cannot resolve packageDirectory from "
                    + resolvedIndexVerification.inspection().indexFile());
        } else {
            try {
                packageVerification = TrainingReportPromotionGateArtifactPackage.verifyComplete(
                        packageDirectory,
                        indexedManifestSha256(resolvedIndexVerification.inspection()).orElse(null),
                        TrainingReportPromotionGateArtifactPackage.Options.defaults());
                failures.addAll(packageVerification.failures());
            } catch (IOException error) {
                failures.add("Complete package verification failed for "
                        + packageDirectory + ": " + error.getMessage());
            }
        }
        return new VerificationIndexPackageAudit(
                resolvedIndexVerification,
                packageVerification,
                failures);
    }

    static VerificationIndexPackageAuditReport writeVerificationIndexPackageAuditReport(
            Path reportFile,
            VerificationIndexPackageAudit audit) throws IOException {
        Path resolvedReportFile = Objects.requireNonNull(reportFile, "reportFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationIndexPackageAudit resolvedAudit = Objects.requireNonNull(audit, "audit must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReportFile,
                TrainerJson.toJson(TrainingReportPromotionGateArtifactPackageMaps.verificationIndexPackageAuditReport(
                        resolvedAudit,
                        Instant.now())) + "\n");
        return new VerificationIndexPackageAuditReport(
                resolvedReportFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedReportFile),
                resolvedAudit);
    }

    static VerificationIndexPackageAuditReport auditVerificationIndexPackageAndWriteReport(
            Path indexFile,
            String expectedIndexSha256,
            Path reportFile) throws IOException {
        VerificationIndexPackageAudit audit =
                TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackage(
                        indexFile,
                        expectedIndexSha256);
        return writeVerificationIndexPackageAuditReport(reportFile, audit);
    }

    static VerificationEvidenceManifest writeVerificationEvidenceManifest(
            Path evidenceFile,
            VerifiedPackageBundle verifiedPackage) throws IOException {
        Path resolvedEvidenceFile = Objects.requireNonNull(evidenceFile, "evidenceFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerifiedPackageBundle resolvedVerifiedPackage = Objects.requireNonNull(
                verifiedPackage,
                "verifiedPackage must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedEvidenceFile,
                TrainerJson.toJson(TrainingReportPromotionGateArtifactPackageMaps.verificationEvidenceManifest(
                        resolvedVerifiedPackage,
                        Instant.now())) + "\n");
        return new VerificationEvidenceManifest(
                resolvedEvidenceFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedEvidenceFile));
    }

    static VerificationEvidenceReceipt writeVerificationEvidenceReceipt(
            Path receiptFile,
            VerificationEvidenceVerification verification) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationEvidenceVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReceiptFile,
                TrainerJson.toJson(TrainingReportPromotionGateArtifactPackageMaps.verificationEvidenceReceipt(
                        resolvedVerification,
                        Instant.now())) + "\n");
        return new VerificationEvidenceReceipt(
                resolvedReceiptFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedReceiptFile),
                resolvedVerification);
    }

    static VerificationEvidenceReceipt verifyVerificationEvidenceManifestAndWriteReceipt(
            Path evidenceFile,
            String expectedEvidenceSha256,
            Path receiptFile) throws IOException {
        VerificationEvidenceVerification verification =
                TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceManifest(
                        evidenceFile,
                        expectedEvidenceSha256);
        return writeVerificationEvidenceReceipt(receiptFile, verification);
    }

    static VerificationIndexReceipt writeVerificationIndexReceipt(
            Path receiptFile,
            VerificationIndexVerification verification) throws IOException {
        Path resolvedReceiptFile = Objects.requireNonNull(receiptFile, "receiptFile must not be null")
                .toAbsolutePath()
                .normalize();
        VerificationIndexVerification resolvedVerification = Objects.requireNonNull(
                verification,
                "verification must not be null");
        TrainerCheckpointIO.writeStringAtomically(
                resolvedReceiptFile,
                TrainerJson.toJson(TrainingReportPromotionGateArtifactPackageMaps.verificationIndexReceipt(
                        resolvedVerification,
                        Instant.now())) + "\n");
        return new VerificationIndexReceipt(
                resolvedReceiptFile,
                TrainingReportPromotionGateArtifactPackageFingerprints.sha256(resolvedReceiptFile),
                resolvedVerification);
    }

    static VerificationIndexReceipt verifyVerificationIndexAndWriteReceipt(
            Path indexFile,
            String expectedIndexSha256,
            Path receiptFile) throws IOException {
        VerificationIndexVerification verification =
                TrainingReportPromotionGateArtifactPackage.verifyVerificationIndex(
                        indexFile,
                        expectedIndexSha256);
        return writeVerificationIndexReceipt(receiptFile, verification);
    }

    private static Optional<String> indexedManifestSha256(VerificationIndexInspection inspection) {
        return TrainingReportPromotionGateMapValues.objectValue(inspection.index(), "manifest")
                .flatMap(manifest -> TrainingReportPromotionGateMapValues.stringValue(manifest, "sha256"));
    }
}
