package tech.kayys.gollek.ml;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import tech.kayys.gollek.ml.train.TrainingReport;
import tech.kayys.gollek.ml.train.TrainingReportPortfolio;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGate;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGateManifest;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGateManifestJUnitXml;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGateManifestJUnitXmlContract;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGateManifestVerificationReport;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGateManifestVerificationReportReceipt;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfile;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCatalog;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfilePromotionGate;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfilePromotionGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileValidationGate;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileValidationGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportReader;
import tech.kayys.gollek.ml.train.TrainingReportValidationMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportValidationPolicy;

/**
 * Quality-profile helpers for trainer report validation and promotion workflows.
 */
public class GollekDlQualityProfileFacade extends GollekDlTrainingFacade {
    protected GollekDlQualityProfileFacade() {
    }

    public static List<TrainingReportQualityProfile> trainingReportQualityProfiles() {
        return TrainingReportQualityProfile.defaults();
    }

    public static TrainingReportQualityProfile trainingReportQualityProfile(String id) {
        return TrainingReportQualityProfile.require(id);
    }

    public static TrainingReportQualityProfileCatalog trainingReportQualityProfileCatalog() {
        return TrainingReportQualityProfileCatalog.defaults();
    }

    public static String trainingReportQualityProfilesJson() {
        return trainingReportQualityProfileCatalog().toJson();
    }

    public static String trainingReportQualityProfilesMarkdown() {
        return TrainingReportQualityProfileMarkdown.render(trainingReportQualityProfileCatalog());
    }

    public static String trainingReportQualityProfilesMarkdown(List<TrainingReportQualityProfile> profiles) {
        return TrainingReportQualityProfileMarkdown.render(profiles);
    }

    public static TrainingReportQualityProfileArtifacts.ArtifactBundle writeTrainingReportQualityProfileArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileArtifacts.write(outputDirectory);
    }

    public static TrainingReportQualityProfileArtifacts.ArtifactBundle writeTrainingReportQualityProfileArtifacts(
            Path outputDirectory,
            TrainingReportQualityProfileCatalog catalog) throws IOException {
        return TrainingReportQualityProfileArtifacts.write(outputDirectory, catalog);
    }

    public static TrainingReportQualityProfileArtifacts.ArtifactBundle writeTrainingReportQualityProfileArtifacts(
            Path outputDirectory,
            List<TrainingReportQualityProfile> profiles) throws IOException {
        return TrainingReportQualityProfileArtifacts.write(
                outputDirectory,
                new TrainingReportQualityProfileCatalog(profiles));
    }

    public static TrainingReportQualityProfileArtifacts.ArtifactBundle refreshTrainingReportQualityProfileArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileArtifacts.refreshDerived(outputDirectory);
    }

    public static TrainingReportQualityProfileArtifacts.ArtifactInspection readTrainingReportQualityProfileArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileArtifacts.read(outputDirectory);
    }

    public static TrainingReportQualityProfileArtifacts.ArtifactVerification verifyTrainingReportQualityProfileArtifacts(
            TrainingReportQualityProfileArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportQualityProfileArtifacts.verify(bundle);
    }

    public static TrainingReportQualityProfileArtifacts.ArtifactVerification verifyTrainingReportQualityProfileArtifacts(
            Path outputDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256) throws IOException {
        return TrainingReportQualityProfileArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256);
    }

    public static TrainingReportValidationPolicy.Result validateTrainingReport(
            Path reportFile,
            TrainingReportQualityProfile profile) throws IOException {
        TrainingReportQualityProfile resolvedProfile = profile == null
                ? TrainingReportQualityProfile.strictCi()
                : profile;
        return TrainingReportReader.readReport(reportFile).validate(resolvedProfile.validationPolicy());
    }

    public static TrainingReportValidationPolicy.Result validateTrainingReport(
            TrainingReport report,
            TrainingReportQualityProfile profile) {
        TrainingReportQualityProfile resolvedProfile = profile == null
                ? TrainingReportQualityProfile.strictCi()
                : profile;
        return report.validate(resolvedProfile.validationPolicy());
    }

    public static String trainingReportValidationMarkdown(
            Path reportFile,
            TrainingReportQualityProfile profile) throws IOException {
        return TrainingReportValidationMarkdown.render(validateTrainingReport(reportFile, profile));
    }

    public static String trainingReportValidationMarkdown(
            TrainingReport report,
            TrainingReportQualityProfile profile) {
        return TrainingReportValidationMarkdown.render(validateTrainingReport(report, profile));
    }

    public static TrainingReportQualityProfileValidationGate.Result runTrainingReportQualityProfileValidationGate(
            TrainingReportQualityProfileValidationGate.Request request) throws IOException {
        return TrainingReportQualityProfileValidationGate.evaluate(request);
    }

    public static TrainingReportQualityProfileValidationGate.Result runTrainingReportQualityProfileValidationGate(
            Path reportFile,
            TrainingReportQualityProfile profile,
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileValidationGate.evaluate(reportFile, profile, outputDirectory);
    }

    public static TrainingReportQualityProfileValidationGate.Result runTrainingReportQualityProfileValidationGate(
            Path reportFile,
            String profileId,
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileValidationGate.evaluateProfile(reportFile, profileId, outputDirectory);
    }

    public static String trainingReportQualityProfileValidationGateMarkdown(
            TrainingReportQualityProfileValidationGate.Result result) {
        return result.markdown();
    }

    public static TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle
            writeTrainingReportQualityProfileValidationGateArtifacts(
                    Path outputDirectory,
                    TrainingReportQualityProfileValidationGate.Result result) throws IOException {
        return TrainingReportQualityProfileValidationGateArtifacts.write(outputDirectory, result);
    }

    public static TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle
            writeTrainingReportQualityProfileValidationGateArtifacts(
                    Path outputDirectory,
                    TrainingReportQualityProfileValidationGate.Result result,
                    TrainingReportQualityProfileValidationGateArtifacts.Options options) throws IOException {
        return TrainingReportQualityProfileValidationGateArtifacts.write(outputDirectory, result, options);
    }

    public static TrainingReportQualityProfileValidationGateArtifacts.ArtifactInspection
            readTrainingReportQualityProfileValidationGateArtifacts(Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileValidationGateArtifacts.read(outputDirectory);
    }

    public static TrainingReportQualityProfileValidationGateArtifacts.ArtifactInspection
            refreshTrainingReportQualityProfileValidationGateArtifacts(Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileValidationGateArtifacts.refreshDerived(outputDirectory);
    }

    public static TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification
            verifyTrainingReportQualityProfileValidationGateArtifacts(
                    TrainingReportQualityProfileValidationGateArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportQualityProfileValidationGateArtifacts.verify(bundle);
    }

    public static TrainingReportQualityProfileValidationGateArtifacts.ArtifactVerification
            verifyTrainingReportQualityProfileValidationGateArtifacts(
                    Path outputDirectory,
                    String expectedJsonSha256,
                    String expectedMarkdownSha256) throws IOException {
        return TrainingReportQualityProfileValidationGateArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256);
    }

    public static TrainingReportPortfolio.PromotionDecision trainingReportPromotionDecision(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportQualityProfile profile) throws IOException {
        TrainingReportQualityProfile resolvedProfile = profile == null
                ? TrainingReportQualityProfile.productionPromotion()
                : profile;
        return TrainingReportPortfolio.fromFiles(reportFiles)
                .promotionDecision(baselineName, resolvedProfile.promotionPolicy());
    }

    public static TrainingReportPortfolio.PromotionReview trainingReportPromotionReview(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportQualityProfile profile) throws IOException {
        TrainingReportQualityProfile resolvedProfile = profile == null
                ? TrainingReportQualityProfile.productionPromotion()
                : profile;
        return TrainingReportPortfolio.fromFiles(reportFiles)
                .promotionReview(baselineName, resolvedProfile.promotionPolicy());
    }

    public static TrainingReportQualityProfilePromotionGate.Result runTrainingReportQualityProfilePromotionGate(
            TrainingReportQualityProfilePromotionGate.Request request) throws IOException {
        return TrainingReportQualityProfilePromotionGate.evaluate(request);
    }

    public static TrainingReportQualityProfilePromotionGate.Result runTrainingReportQualityProfilePromotionGate(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportQualityProfile profile,
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfilePromotionGate.evaluate(
                reportFiles,
                baselineName,
                profile,
                outputDirectory);
    }

    public static TrainingReportQualityProfilePromotionGate.Result runTrainingReportQualityProfilePromotionGate(
            Map<String, Path> reportFiles,
            String baselineName,
            String profileId,
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfilePromotionGate.evaluateProfile(
                reportFiles,
                baselineName,
                profileId,
                outputDirectory);
    }

    public static TrainingReportQualityProfileCiGate.Result runTrainingReportQualityProfileCiGate(
            TrainingReportQualityProfileCiGate.Request request) throws IOException {
        return TrainingReportQualityProfileCiGate.evaluate(request);
    }

    public static TrainingReportQualityProfileCiGate.Result runTrainingReportQualityProfileCiGate(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportQualityProfile profile,
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileCiGate.evaluate(
                reportFiles,
                baselineName,
                profile,
                outputDirectory);
    }

    public static TrainingReportQualityProfileCiGate.Result runTrainingReportQualityProfileCiGate(
            Map<String, Path> reportFiles,
            String baselineName,
            String profileId,
            Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileCiGate.evaluateProfile(
                reportFiles,
                baselineName,
                profileId,
                outputDirectory);
    }

    public static String trainingReportQualityProfileCiGateMarkdown(
            TrainingReportQualityProfileCiGate.Result result) {
        return result.markdown();
    }

    public static TrainingReportQualityProfileCiGateManifest.ManifestBundle
            writeTrainingReportQualityProfileCiGateManifest(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGate.Result result) throws IOException {
        return TrainingReportQualityProfileCiGateManifest.write(outputDirectory, result);
    }

    public static TrainingReportQualityProfileCiGateManifest.ManifestBundle
            writeTrainingReportQualityProfileCiGateManifest(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGate.Result result,
                    TrainingReportQualityProfileCiGateManifest.Options options) throws IOException {
        return TrainingReportQualityProfileCiGateManifest.write(outputDirectory, result, options);
    }

    public static TrainingReportQualityProfileCiGateManifest.ManifestInspection
            readTrainingReportQualityProfileCiGateManifest(Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileCiGateManifest.read(outputDirectory);
    }

    public static TrainingReportQualityProfileCiGateManifest.ManifestInspection
            refreshTrainingReportQualityProfileCiGateManifest(Path outputDirectory) throws IOException {
        return TrainingReportQualityProfileCiGateManifest.refreshDerived(outputDirectory);
    }

    public static TrainingReportQualityProfileCiGateManifest.ManifestVerification
            verifyTrainingReportQualityProfileCiGateManifest(
                    TrainingReportQualityProfileCiGateManifest.ManifestBundle bundle) throws IOException {
        return TrainingReportQualityProfileCiGateManifest.verify(bundle);
    }

    public static TrainingReportQualityProfileCiGateManifest.ManifestVerification
            verifyTrainingReportQualityProfileCiGateManifest(
                    Path outputDirectory,
                    String expectedJsonSha256,
                    String expectedMarkdownSha256) throws IOException {
        return TrainingReportQualityProfileCiGateManifest.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256);
    }

    public static String trainingReportQualityProfileCiGateManifestJUnitXml(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        return TrainingReportQualityProfileCiGateManifestJUnitXml.render(verification);
    }

    public static TrainingReportQualityProfileCiGateManifestJUnitXml.Report
            writeTrainingReportQualityProfileCiGateManifestJUnitXml(
                    Path junitXmlFile,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        return TrainingReportQualityProfileCiGateManifestJUnitXml.write(junitXmlFile, verification);
    }

    public static TrainingReportQualityProfileCiGateManifestJUnitXmlContract.Inspection
            inspectTrainingReportQualityProfileCiGateManifestJUnitXml(
                    String junitXml,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        return TrainingReportQualityProfileCiGateManifestJUnitXmlContract.inspect(junitXml, verification);
    }

    public static String trainingReportQualityProfileCiGateManifestVerificationJson(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.renderJson(verification);
    }

    public static String trainingReportQualityProfileCiGateManifestVerificationMarkdown(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.renderMarkdown(verification);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReport.ReportBundle
            writeTrainingReportQualityProfileCiGateManifestVerificationReport(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.write(outputDirectory, verification);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReport.ReportBundle
            writeTrainingReportQualityProfileCiGateManifestVerificationReport(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification,
                    TrainingReportQualityProfileCiGateManifestVerificationReport.Options options)
                    throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.write(
                outputDirectory,
                verification,
                options);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReport.ReportInspection
            readTrainingReportQualityProfileCiGateManifestVerificationReport(Path outputDirectory)
                    throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.read(outputDirectory);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReport.ReportInspection
            readTrainingReportQualityProfileCiGateManifestVerificationReport(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGateManifestVerificationReport.Options options)
                    throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.read(outputDirectory, options);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification
            verifyTrainingReportQualityProfileCiGateManifestVerificationReport(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.verify(outputDirectory, verification);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification
            verifyTrainingReportQualityProfileCiGateManifestVerificationReport(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification,
                    TrainingReportQualityProfileCiGateManifestVerificationReport.Options options)
                    throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReport.verify(
                outputDirectory,
                verification,
                options);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.Receipt
            writeTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                    Path receiptFile,
                    TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification verification)
                    throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.write(receiptFile, verification);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.Receipt
            writeTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                    Path outputDirectory,
                    TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification verification,
                    String receiptFileName) throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.write(
                outputDirectory,
                verification,
                receiptFileName);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptInspection
            readTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(Path receiptFile)
                    throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.read(receiptFile);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptVerification
            verifyTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                    Path receiptFile,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.verify(receiptFile, verification);
    }

    public static TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.ReceiptVerification
            verifyTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                    Path receiptFile,
                    String expectedReceiptSha256,
                    TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        return TrainingReportQualityProfileCiGateManifestVerificationReportReceipt.verify(
                receiptFile,
                expectedReceiptSha256,
                verification);
    }

    public static String trainingReportQualityProfilePromotionGateMarkdown(
            TrainingReportQualityProfilePromotionGate.Result result) {
        return result.markdown();
    }

    public static TrainingReportQualityProfilePromotionGateArtifacts.ArtifactBundle
            writeTrainingReportQualityProfilePromotionGateArtifacts(
                    Path outputDirectory,
                    TrainingReportQualityProfilePromotionGate.Result result) throws IOException {
        return TrainingReportQualityProfilePromotionGateArtifacts.write(outputDirectory, result);
    }

    public static TrainingReportQualityProfilePromotionGateArtifacts.ArtifactBundle
            writeTrainingReportQualityProfilePromotionGateArtifacts(
                    Path outputDirectory,
                    TrainingReportQualityProfilePromotionGate.Result result,
                    TrainingReportQualityProfilePromotionGateArtifacts.Options options) throws IOException {
        return TrainingReportQualityProfilePromotionGateArtifacts.write(outputDirectory, result, options);
    }

    public static TrainingReportQualityProfilePromotionGateArtifacts.ArtifactInspection
            readTrainingReportQualityProfilePromotionGateArtifacts(Path outputDirectory) throws IOException {
        return TrainingReportQualityProfilePromotionGateArtifacts.read(outputDirectory);
    }

    public static TrainingReportQualityProfilePromotionGateArtifacts.ArtifactInspection
            refreshTrainingReportQualityProfilePromotionGateArtifacts(Path outputDirectory) throws IOException {
        return TrainingReportQualityProfilePromotionGateArtifacts.refreshDerived(outputDirectory);
    }

    public static TrainingReportQualityProfilePromotionGateArtifacts.ArtifactVerification
            verifyTrainingReportQualityProfilePromotionGateArtifacts(
                    TrainingReportQualityProfilePromotionGateArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportQualityProfilePromotionGateArtifacts.verify(bundle);
    }

    public static TrainingReportQualityProfilePromotionGateArtifacts.ArtifactVerification
            verifyTrainingReportQualityProfilePromotionGateArtifacts(
                    Path outputDirectory,
                    String expectedJsonSha256,
                    String expectedMarkdownSha256) throws IOException {
        return TrainingReportQualityProfilePromotionGateArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256);
    }
}
