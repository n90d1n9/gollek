package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.Options;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageBundle;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageInspection;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageRefresh;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationEvidenceManifest;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceipt;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndex;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReport;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationIndexReceipt;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundle;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceipt;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle;

final class TrainingReportPromotionGatePackageWorkflow {
    private static final String GATE_JSON_ARTIFACT = "json";
    private static final String GATE_MARKDOWN_ARTIFACT = "markdown";
    private static final String GATE_JUNIT_XML_ARTIFACT = "junitXml";

    private TrainingReportPromotionGatePackageWorkflow() {
    }

    static PackageBundle write(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportPromotionGate.Result resolvedResult =
                Objects.requireNonNull(result, "result must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");

        TrainingReportPromotionArtifacts.ArtifactBundle review = TrainingReportPromotionArtifacts.write(
                resolvedDirectory,
                resolvedResult.review(),
                resolvedOptions.review());
        TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts =
                TrainingReportPromotionGateArtifacts.write(
                        resolvedDirectory,
                        resolvedResult,
                        resolvedOptions.artifacts());
        Map<String, Path> packageArtifacts = TrainingReportPromotionGatePackageSupport.packageArtifactPaths(
                review,
                TrainingReportPromotionGateSourceSnapshots.snapshotFiles(resolvedDirectory, review));
        TrainingReportPromotionGateArtifactManifest.ManifestBundle manifest =
                TrainingReportPromotionGateArtifactManifest.write(
                        artifacts,
                        packageArtifacts,
                        resolvedOptions.manifest(),
                        resolvedGeneratedAt);
        return new PackageBundle(resolvedDirectory, review, artifacts, manifest);
    }

    static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPromotionGate.Result result,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        PackageBundle bundle = write(directory, result, options, generatedAt);
        return verifyPackage(bundle, reportDirectory);
    }

    static PackageBundle run(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedOutputDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        TrainingReportPromotionGate.Request request = new TrainingReportPromotionGate.Request(
                reportFiles,
                baselineName,
                policy,
                resolvedOutputDirectory,
                resolvedOptions.review());
        TrainingReportPromotionGate.Result result = TrainingReportPromotionGate.evaluate(request);
        return write(resolvedOutputDirectory, result, resolvedOptions, resolvedGeneratedAt);
    }

    static VerifiedPackageBundle runAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        PackageBundle bundle = run(reportFiles, baselineName, policy, outputDirectory, options, generatedAt);
        return verifyPackage(bundle, reportDirectory);
    }

    static PackageBundle runWithSeverity(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedOutputDirectory = Objects.requireNonNull(outputDirectory, "outputDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportDiagnostics.Severity resolvedSeverity = Objects.requireNonNull(
                maxAllowedDiagnosticSeverity,
                "maxAllowedDiagnosticSeverity must not be null");
        TrainingReportPortfolio.PromotionPolicy policy = TrainingReportPortfolio.PromotionPolicy.defaultPolicy()
                .withMaxCandidateDiagnosticSeverity(resolvedSeverity);
        return run(
                reportFiles,
                baselineName,
                policy,
                resolvedOutputDirectory,
                options,
                generatedAt);
    }

    static VerifiedPackageBundle runWithSeverityAndVerify(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        PackageBundle bundle = runWithSeverity(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                options,
                generatedAt);
        return verifyPackage(bundle, reportDirectory);
    }

    static PackageInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest =
                TrainingReportPromotionGateArtifactManifest.read(
                        resolvedDirectory,
                        resolvedOptions.manifest());
        return TrainingReportPromotionGatePackageSupport.readFromManifest(manifest, resolvedOptions);
    }

    static TrainingReportPromotionGateArtifactManifest.ManifestVerification verify(PackageBundle bundle)
            throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return TrainingReportPromotionGateArtifactManifest.verify(bundle.manifest());
    }

    static TrainingReportPromotionGateArtifactManifest.ManifestVerification verify(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return TrainingReportPromotionGateArtifactManifest.verify(
                resolvedDirectory,
                expectedManifestSha256,
                resolvedOptions.manifest());
    }

    static PackageVerification verifyComplete(PackageBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestVerification = verify(bundle);
        PackageInspection inspection = TrainingReportPromotionGatePackageSupport.readFromManifest(
                manifestVerification.inspection(),
                Options.defaults());
        SourceSnapshotVerification sourceSnapshotVerification = TrainingReportPromotionGateSourceSnapshots.verify(
                inspection);
        return TrainingReportPromotionGatePackageSupport.packageVerification(
                inspection,
                manifestVerification,
                sourceSnapshotVerification);
    }

    static VerifiedPackageBundle verifyPackage(PackageBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verifyPackage(bundle, bundle.directory());
    }

    static VerifiedPackageBundle verifyPackage(PackageBundle bundle, Path reportDirectory) throws IOException {
        PackageBundle resolvedBundle = Objects.requireNonNull(bundle, "bundle must not be null");
        PackageVerification verification = verifyComplete(resolvedBundle);
        VerificationReportBundle reports =
                TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationReports(
                        reportDirectory,
                        verification);
        VerificationReportBundleReceipt reportBundleReceipt =
                TrainingReportPromotionGatePackageVerificationArtifacts.verifyVerificationReportBundleAndWriteReceipt(
                        reports.directory(),
                        reports.json().reportSha256(),
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationReportBundleReceiptFile(
                                reports.directory()));
        VerifiedPackageBundle verifiedPackage = new VerifiedPackageBundle(
                resolvedBundle,
                reports,
                reportBundleReceipt);
        VerificationIndex index = TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationIndex(
                TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexFile(reports.directory()),
                verifiedPackage);
        VerificationIndexReceipt receipt =
                TrainingReportPromotionGatePackageVerificationArtifacts.verifyVerificationIndexAndWriteReceipt(
                        index.indexFile(),
                        index.indexSha256(),
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexReceiptFile(
                                reports.directory()));
        VerificationIndexPackageAuditReport packageAuditReport =
                TrainingReportPromotionGatePackageVerificationArtifacts.auditVerificationIndexPackageAndWriteReport(
                        index.indexFile(),
                        index.indexSha256(),
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationIndexPackageAuditFile(
                                reports.directory()));
        VerifiedPackageBundle verifiedPackageWithAudit =
                new VerifiedPackageBundle(resolvedBundle, reports, reportBundleReceipt, index, receipt,
                        packageAuditReport);
        VerificationEvidenceManifest evidence =
                TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationEvidenceManifest(
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationEvidenceFile(
                                reports.directory()),
                        verifiedPackageWithAudit);
        VerificationEvidenceReceipt evidenceReceipt =
                TrainingReportPromotionGatePackageVerificationArtifacts.verifyVerificationEvidenceManifestAndWriteReceipt(
                        evidence.evidenceFile(),
                        evidence.evidenceSha256(),
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationEvidenceReceiptFile(
                                reports.directory()));
        return new VerifiedPackageBundle(
                resolvedBundle,
                reports,
                reportBundleReceipt,
                index,
                receipt,
                packageAuditReport,
                evidence,
                evidenceReceipt);
    }

    static PackageRefresh refreshComplete(
            Path directory,
            Options options,
            Instant generatedAt,
            Path reportDirectory) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        Path resolvedReportDirectory = Objects.requireNonNull(reportDirectory, "reportDirectory must not be null")
                .toAbsolutePath()
                .normalize();

        TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest =
                TrainingReportPromotionGateArtifactManifest.read(
                        resolvedDirectory,
                        resolvedOptions.manifest());
        TrainingReportPromotionGatePackageSupport.verifyNonRefreshableManifestArtifacts(
                manifest,
                Set.of(GATE_MARKDOWN_ARTIFACT, GATE_JUNIT_XML_ARTIFACT));
        TrainingReportPromotionGateArtifacts.ArtifactInspection artifacts =
                TrainingReportPromotionGateArtifacts.refreshDerivedFiles(
                        TrainingReportPromotionGatePackageSupport.requiredManifestArtifact(
                                manifest,
                                GATE_JSON_ARTIFACT).file(),
                        TrainingReportPromotionGatePackageSupport.requiredManifestArtifact(
                                manifest,
                                GATE_MARKDOWN_ARTIFACT).file(),
                        TrainingReportPromotionGatePackageSupport.requiredManifestArtifact(
                                manifest,
                                GATE_JUNIT_XML_ARTIFACT).file());
        TrainingReportPromotionGateArtifactManifest.ManifestInspection refreshedManifest =
                TrainingReportPromotionGateArtifactManifest.refresh(manifest, resolvedGeneratedAt);
        PackageVerification verification = verifyComplete(resolvedDirectory, null, resolvedOptions);
        VerificationReportBundle reports =
                TrainingReportPromotionGatePackageVerificationArtifacts.writeVerificationReports(
                        resolvedReportDirectory,
                        verification);
        VerificationReportBundleVerification reportVerification =
                TrainingReportPromotionGateVerificationReportBundleVerifier.verify(
                        resolvedReportDirectory,
                        reports.json().reportSha256());
        return new PackageRefresh(
                resolvedDirectory,
                refreshedManifest,
                artifacts,
                verification,
                reports,
                reportVerification);
    }

    static PackageVerification verifyComplete(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestVerification =
                TrainingReportPromotionGateArtifactManifest.verify(
                        resolvedDirectory,
                        expectedManifestSha256,
                        resolvedOptions.manifest());
        PackageInspection inspection = TrainingReportPromotionGatePackageSupport.readFromManifest(
                manifestVerification.inspection(),
                resolvedOptions);
        SourceSnapshotVerification sourceSnapshotVerification = TrainingReportPromotionGateSourceSnapshots.verify(
                inspection);
        return TrainingReportPromotionGatePackageSupport.packageVerification(
                inspection,
                manifestVerification,
                sourceSnapshotVerification);
    }
}
