package tech.kayys.gollek.ml;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import tech.kayys.gollek.ml.train.TrainerRuntimeSmoke;
import tech.kayys.gollek.ml.train.TrainerRuntimeSmokeArtifacts;
import tech.kayys.gollek.ml.train.TrainingReport;
import tech.kayys.gollek.ml.train.TrainingReportAcceleration;
import tech.kayys.gollek.ml.train.TrainingReportActionPlan;
import tech.kayys.gollek.ml.train.TrainingReportActionPlanMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportAdvisor;
import tech.kayys.gollek.ml.train.TrainingReportComparison;
import tech.kayys.gollek.ml.train.TrainingReportComparisonActionPlanArtifactVerificationMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportComparisonActionPlanArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportComparisonActionPlanExport;
import tech.kayys.gollek.ml.train.TrainingReportComparisonArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportComparisonExport;
import tech.kayys.gollek.ml.train.TrainingReportComparisonJUnitXml;
import tech.kayys.gollek.ml.train.TrainingReportComparisonMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportDiagnostics;
import tech.kayys.gollek.ml.train.TrainingReportEpochSnapshot;
import tech.kayys.gollek.ml.train.TrainingReportHistoryOverview;
import tech.kayys.gollek.ml.train.TrainingReportPortfolio;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactPackage;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactPackageReport;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifactManifest;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioExport;
import tech.kayys.gollek.ml.train.TrainingReportPortfolioMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportPromotionArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGate;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactManifest;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateJUnitXml;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportPromotionMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportRecommendation;
import tech.kayys.gollek.ml.train.TrainingReportReader;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGate;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileActionPlan;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileBalanceAssessment;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileBudgetGate;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileBudgetGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfile;
import tech.kayys.gollek.ml.train.TrainingReportSeries;
import tech.kayys.gollek.ml.train.TrainingReportSeriesBundle;
import tech.kayys.gollek.ml.train.TrainingReportSeriesExport;
import tech.kayys.gollek.ml.train.TrainingReportThroughput;
import tech.kayys.gollek.ml.train.TrainingReportValidationArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportValidationJUnitXml;
import tech.kayys.gollek.ml.train.TrainingReportValidationMarkdown;
import tech.kayys.gollek.ml.train.TrainingReportValidationPolicy;

/**
 * Public one-call training/evaluation facade inherited by {@link Gollek.DL}.
 */
@SuppressWarnings("deprecation")
public class GollekDlTrainingFacade extends GollekDlFitFacade {
    protected GollekDlTrainingFacade() {
    }

    public static TrainerRuntimeSmoke.Result trainerRuntimeSmoke() throws IOException {
        return TrainerRuntimeSmoke.run();
    }

    public static TrainerRuntimeSmoke.Result trainerRuntimeSmoke(Path checkpointDir) throws IOException {
        return TrainerRuntimeSmoke.run(checkpointDir);
    }

    public static TrainerRuntimeSmoke.Result trainerRuntimeSmoke(
            TrainerRuntimeSmoke.Options options) throws IOException {
        return TrainerRuntimeSmoke.run(options);
    }

    public static TrainerRuntimeSmokeArtifacts.ArtifactBundle writeTrainerRuntimeSmokeArtifacts(
            Path outputDirectory,
            TrainerRuntimeSmoke.Result result) throws IOException {
        return TrainerRuntimeSmokeArtifacts.write(outputDirectory, result);
    }

    public static TrainerRuntimeSmokeArtifacts.ArtifactBundle refreshTrainerRuntimeSmokeArtifacts(
            Path outputDirectory) throws IOException {
        return TrainerRuntimeSmokeArtifacts.refresh(outputDirectory);
    }

    public static TrainerRuntimeSmokeArtifacts.ArtifactInspection readTrainerRuntimeSmokeArtifacts(
            Path outputDirectory) throws IOException {
        return TrainerRuntimeSmokeArtifacts.read(outputDirectory);
    }

    public static TrainerRuntimeSmokeArtifacts.ArtifactVerification verifyTrainerRuntimeSmokeArtifacts(
            TrainerRuntimeSmokeArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainerRuntimeSmokeArtifacts.verify(bundle);
    }

    public static TrainerRuntimeSmokeArtifacts.ArtifactVerification verifyTrainerRuntimeSmokeArtifacts(
            Path outputDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) throws IOException {
        return TrainerRuntimeSmokeArtifacts.verify(
                TrainerRuntimeSmokeArtifacts.read(outputDirectory),
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256);
    }

    public static Map<String, Object> readTrainingReport(Path reportFile) throws IOException {
        return TrainingReportReader.readCanonical(reportFile);
    }

    public static TrainingReport trainingReport(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile);
    }

    public static TrainingReportThroughput trainingReportThroughput(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).throughput();
    }

    public static TrainingReportThroughput trainingReportThroughput(TrainingReport report) {
        return report.throughput();
    }

    public static Map<String, Object> trainingReportThroughputMap(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).throughputMap();
    }

    public static Map<String, Object> trainingReportThroughputMap(TrainingReport report) {
        return report.throughputMap();
    }

    public static String trainingReportThroughputMarkdown(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).throughputMarkdown();
    }

    public static String trainingReportThroughputMarkdown(TrainingReport report) {
        return report.throughputMarkdown();
    }

    public static TrainingReportAcceleration trainingReportAcceleration(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).acceleration();
    }

    public static TrainingReportAcceleration trainingReportAcceleration(TrainingReport report) {
        return report.acceleration();
    }

    public static Map<String, Object> trainingReportAccelerationMap(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).accelerationMap();
    }

    public static Map<String, Object> trainingReportAccelerationMap(TrainingReport report) {
        return report.accelerationMap();
    }

    public static String trainingReportAccelerationMarkdown(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).accelerationMarkdown();
    }

    public static String trainingReportAccelerationMarkdown(TrainingReport report) {
        return report.accelerationMarkdown();
    }

    public static TrainingReportActionPlan trainingReportActionPlan(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).actionPlan();
    }

    public static TrainingReportActionPlan trainingReportActionPlan(TrainingReport report) {
        return report.actionPlan();
    }

    public static String trainingReportActionPlanMarkdown(Path reportFile) throws IOException {
        return TrainingReportActionPlanMarkdown.render(TrainingReportReader.readReport(reportFile));
    }

    public static String trainingReportActionPlanMarkdown(TrainingReport report) {
        return TrainingReportActionPlanMarkdown.render(report);
    }

    public static String trainingReportActionPlanMarkdown(TrainingReportActionPlan actionPlan) {
        return TrainingReportActionPlanMarkdown.render(actionPlan);
    }

    public static Map<String, Object> trainingReportRuntimeInputProfile(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeInputProfileMap();
    }

    public static Map<String, Object> trainingReportRuntimeInputProfile(TrainingReport report) {
        return report.runtimeInputProfileMap();
    }

    public static String trainingReportRuntimeInputProfileMarkdown(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeInputProfileMarkdown();
    }

    public static String trainingReportRuntimeInputProfileMarkdown(TrainingReport report) {
        return report.runtimeInputProfileMarkdown();
    }

    public static TrainingReportRuntimeProfile.Balance trainingReportRuntimeProfileBalance(Path reportFile)
            throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBalance();
    }

    public static TrainingReportRuntimeProfile.Balance trainingReportRuntimeProfileBalance(TrainingReport report) {
        return report.runtimeProfileBalance();
    }

    public static Map<String, Object> trainingReportRuntimeProfileBalanceMap(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBalanceMap();
    }

    public static Map<String, Object> trainingReportRuntimeProfileBalanceMap(TrainingReport report) {
        return report.runtimeProfileBalanceMap();
    }

    public static TrainingReportRuntimeProfileBalanceAssessment trainingReportRuntimeProfileBalanceAssessment(
            Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBalanceAssessment();
    }

    public static TrainingReportRuntimeProfileBalanceAssessment trainingReportRuntimeProfileBalanceAssessment(
            Path reportFile,
            TrainingReportRuntimeProfileBalanceAssessment.Thresholds thresholds) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBalanceAssessment(thresholds);
    }

    public static TrainingReportRuntimeProfileBalanceAssessment trainingReportRuntimeProfileBalanceAssessment(
            TrainingReport report) {
        return report.runtimeProfileBalanceAssessment();
    }

    public static TrainingReportRuntimeProfileBalanceAssessment trainingReportRuntimeProfileBalanceAssessment(
            TrainingReport report,
            TrainingReportRuntimeProfileBalanceAssessment.Thresholds thresholds) {
        return report.runtimeProfileBalanceAssessment(thresholds);
    }

    public static Map<String, Object> trainingReportRuntimeProfileBalanceAssessmentMap(Path reportFile)
            throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBalanceAssessmentMap();
    }

    public static Map<String, Object> trainingReportRuntimeProfileBalanceAssessmentMap(TrainingReport report) {
        return report.runtimeProfileBalanceAssessmentMap();
    }

    public static String trainingReportRuntimeProfileBalanceMarkdown(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBalanceMarkdown();
    }

    public static String trainingReportRuntimeProfileBalanceMarkdown(TrainingReport report) {
        return report.runtimeProfileBalanceMarkdown();
    }

    public static TrainingReportRuntimeProfileActionPlan trainingReportRuntimeProfileActionPlan(Path reportFile)
            throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileActionPlan();
    }

    public static TrainingReportRuntimeProfileActionPlan trainingReportRuntimeProfileActionPlan(TrainingReport report) {
        return report.runtimeProfileActionPlan();
    }

    public static Map<String, Object> trainingReportRuntimeProfileActionPlanMap(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileActionPlanMap();
    }

    public static Map<String, Object> trainingReportRuntimeProfileActionPlanMap(TrainingReport report) {
        return report.runtimeProfileActionPlanMap();
    }

    public static String trainingReportRuntimeProfileActionPlanMarkdown(Path reportFile) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileActionPlanMarkdown();
    }

    public static String trainingReportRuntimeProfileActionPlanMarkdown(TrainingReport report) {
        return report.runtimeProfileActionPlanMarkdown();
    }

    public static TrainingReportRuntimeProfileBudgetGate.Result trainingReportRuntimeProfileBudgetGate(Path reportFile)
            throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBudgetGate();
    }

    public static TrainingReportRuntimeProfileBudgetGate.Result trainingReportRuntimeProfileBudgetGate(
            Path reportFile,
            TrainingReportRuntimeProfileBudgetGate.Policy policy) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeProfileBudgetGate(policy);
    }

    public static TrainingReportRuntimeProfileBudgetGate.Result trainingReportRuntimeProfileBudgetGate(
            TrainingReport report) {
        return report.runtimeProfileBudgetGate();
    }

    public static TrainingReportRuntimeProfileBudgetGate.Result trainingReportRuntimeProfileBudgetGate(
            TrainingReport report,
            TrainingReportRuntimeProfileBudgetGate.Policy policy) {
        return report.runtimeProfileBudgetGate(policy);
    }

    public static String trainingReportRuntimeProfileBudgetGateMarkdown(Path reportFile) throws IOException {
        return trainingReportRuntimeProfileBudgetGate(reportFile).markdown();
    }

    public static String trainingReportRuntimeProfileBudgetGateMarkdown(
            Path reportFile,
            TrainingReportRuntimeProfileBudgetGate.Policy policy) throws IOException {
        return trainingReportRuntimeProfileBudgetGate(reportFile, policy).markdown();
    }

    public static String trainingReportRuntimeProfileBudgetGateMarkdown(TrainingReport report) {
        return report.runtimeProfileBudgetGate().markdown();
    }

    public static String trainingReportRuntimeProfileBudgetGateMarkdown(
            TrainingReport report,
            TrainingReportRuntimeProfileBudgetGate.Policy policy) {
        return report.runtimeProfileBudgetGate(policy).markdown();
    }

    public static String trainingReportRuntimeProfileBudgetGateJUnitXml(Path reportFile) throws IOException {
        return trainingReportRuntimeProfileBudgetGate(reportFile).junitXml();
    }

    public static String trainingReportRuntimeProfileBudgetGateJUnitXml(
            Path reportFile,
            TrainingReportRuntimeProfileBudgetGate.Policy policy) throws IOException {
        return trainingReportRuntimeProfileBudgetGate(reportFile, policy).junitXml();
    }

    public static String trainingReportRuntimeProfileBudgetGateJUnitXml(TrainingReport report) {
        return report.runtimeProfileBudgetGate().junitXml();
    }

    public static String trainingReportRuntimeProfileBudgetGateJUnitXml(
            TrainingReport report,
            TrainingReportRuntimeProfileBudgetGate.Policy policy) {
        return report.runtimeProfileBudgetGate(policy).junitXml();
    }

    public static TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle writeTrainingReportRuntimeProfileBudgetGateArtifacts(
            Path directory,
            TrainingReportRuntimeProfileBudgetGate.Result result) throws IOException {
        return TrainingReportRuntimeProfileBudgetGateArtifacts.write(directory, result);
    }

    public static TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle writeTrainingReportRuntimeProfileBudgetGateArtifacts(
            Path directory,
            TrainingReport report) throws IOException {
        return TrainingReportRuntimeProfileBudgetGateArtifacts.write(directory, report.runtimeProfileBudgetGate());
    }

    public static TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactInspection readTrainingReportRuntimeProfileBudgetGateArtifacts(
            Path directory) throws IOException {
        return TrainingReportRuntimeProfileBudgetGateArtifacts.read(directory);
    }

    public static TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle refreshTrainingReportRuntimeProfileBudgetGateArtifacts(
            Path directory) throws IOException {
        return TrainingReportRuntimeProfileBudgetGateArtifacts.refreshDerived(directory);
    }

    public static TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification verifyTrainingReportRuntimeProfileBudgetGateArtifacts(
            TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportRuntimeProfileBudgetGateArtifacts.verify(bundle);
    }

    public static TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification verifyTrainingReportRuntimeProfileBudgetGateArtifacts(
            Path directory) throws IOException {
        return TrainingReportRuntimeProfileBudgetGateArtifacts.verify(directory);
    }

    public static TrainingReportRuntimeInputProfileGate.Result trainingReportRuntimeInputProfileGate(Path reportFile)
            throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeInputProfileGate();
    }

    public static TrainingReportRuntimeInputProfileGate.Result trainingReportRuntimeInputProfileGate(
            Path reportFile,
            TrainingReportRuntimeInputProfileGate.Policy policy) throws IOException {
        return TrainingReportReader.readReport(reportFile).runtimeInputProfileGate(policy);
    }

    public static TrainingReportRuntimeInputProfileGate.Result trainingReportRuntimeInputProfileGate(
            TrainingReport report) {
        return report.runtimeInputProfileGate();
    }

    public static TrainingReportRuntimeInputProfileGate.Result trainingReportRuntimeInputProfileGate(
            TrainingReport report,
            TrainingReportRuntimeInputProfileGate.Policy policy) {
        return report.runtimeInputProfileGate(policy);
    }

    public static String trainingReportRuntimeInputProfileGateMarkdown(Path reportFile) throws IOException {
        return trainingReportRuntimeInputProfileGate(reportFile).markdown();
    }

    public static String trainingReportRuntimeInputProfileGateMarkdown(
            Path reportFile,
            TrainingReportRuntimeInputProfileGate.Policy policy) throws IOException {
        return trainingReportRuntimeInputProfileGate(reportFile, policy).markdown();
    }

    public static String trainingReportRuntimeInputProfileGateMarkdown(TrainingReport report) {
        return report.runtimeInputProfileGate().markdown();
    }

    public static String trainingReportRuntimeInputProfileGateMarkdown(
            TrainingReport report,
            TrainingReportRuntimeInputProfileGate.Policy policy) {
        return report.runtimeInputProfileGate(policy).markdown();
    }

    public static String trainingReportRuntimeInputProfileGateJUnitXml(Path reportFile) throws IOException {
        return trainingReportRuntimeInputProfileGate(reportFile).junitXml();
    }

    public static String trainingReportRuntimeInputProfileGateJUnitXml(
            Path reportFile,
            TrainingReportRuntimeInputProfileGate.Policy policy) throws IOException {
        return trainingReportRuntimeInputProfileGate(reportFile, policy).junitXml();
    }

    public static String trainingReportRuntimeInputProfileGateJUnitXml(TrainingReport report) {
        return report.runtimeInputProfileGate().junitXml();
    }

    public static String trainingReportRuntimeInputProfileGateJUnitXml(
            TrainingReport report,
            TrainingReportRuntimeInputProfileGate.Policy policy) {
        return report.runtimeInputProfileGate(policy).junitXml();
    }

    public static TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle writeTrainingReportRuntimeInputProfileGateArtifacts(
            Path directory,
            TrainingReportRuntimeInputProfileGate.Result result) throws IOException {
        return TrainingReportRuntimeInputProfileGateArtifacts.write(directory, result);
    }

    public static TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle writeTrainingReportRuntimeInputProfileGateArtifacts(
            Path directory,
            TrainingReport report) throws IOException {
        return TrainingReportRuntimeInputProfileGateArtifacts.write(directory, report.runtimeInputProfileGate());
    }

    public static TrainingReportRuntimeInputProfileGateArtifacts.ArtifactInspection readTrainingReportRuntimeInputProfileGateArtifacts(
            Path directory) throws IOException {
        return TrainingReportRuntimeInputProfileGateArtifacts.read(directory);
    }

    public static TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle refreshTrainingReportRuntimeInputProfileGateArtifacts(
            Path directory) throws IOException {
        return TrainingReportRuntimeInputProfileGateArtifacts.refreshDerived(directory);
    }

    public static TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verifyTrainingReportRuntimeInputProfileGateArtifacts(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportRuntimeInputProfileGateArtifacts.verify(bundle);
    }

    public static TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verifyTrainingReportRuntimeInputProfileGateArtifacts(
            Path directory) throws IOException {
        return TrainingReportRuntimeInputProfileGateArtifacts.verify(directory);
    }

    public static List<TrainingReportRecommendation> trainingReportRecommendations(Path reportFile)
            throws IOException {
        return TrainingReportReader.readReport(reportFile).recommendations();
    }

    public static List<TrainingReportRecommendation> trainingReportRecommendations(TrainingReport report) {
        return report.recommendations();
    }

    public static TrainingReportComparison compareTrainingReports(
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportComparison.compare(
                TrainingReportReader.readReport(baselineReportFile),
                TrainingReportReader.readReport(candidateReportFile));
    }

    public static TrainingReportComparison compareTrainingReports(
            TrainingReport baseline,
            TrainingReport candidate) {
        return TrainingReportComparison.compare(baseline, candidate);
    }

    public static TrainingReportComparisonExport trainingReportComparisonExport(
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return compareTrainingReports(baselineReportFile, candidateReportFile).export();
    }

    public static TrainingReportComparisonExport trainingReportComparisonExport(
            TrainingReport baseline,
            TrainingReport candidate) {
        return compareTrainingReports(baseline, candidate).export();
    }

    public static TrainingReportComparisonExport trainingReportComparisonExport(
            TrainingReportComparison comparison) {
        return comparison.export();
    }

    public static String trainingReportComparisonMarkdown(
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportComparisonMarkdown.render(compareTrainingReports(baselineReportFile, candidateReportFile));
    }

    public static String trainingReportComparisonMarkdown(
            TrainingReport baseline,
            TrainingReport candidate) {
        return TrainingReportComparisonMarkdown.render(compareTrainingReports(baseline, candidate));
    }

    public static String trainingReportComparisonMarkdown(
            TrainingReportComparison comparison) {
        return TrainingReportComparisonMarkdown.render(comparison);
    }

    public static String trainingReportComparisonMarkdown(
            TrainingReportComparisonExport export) {
        return TrainingReportComparisonMarkdown.render(export);
    }

    public static String trainingReportComparisonJUnitXml(
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportComparisonJUnitXml.render(compareTrainingReports(baselineReportFile, candidateReportFile));
    }

    public static String trainingReportComparisonJUnitXml(
            TrainingReport baseline,
            TrainingReport candidate) {
        return TrainingReportComparisonJUnitXml.render(compareTrainingReports(baseline, candidate));
    }

    public static String trainingReportComparisonJUnitXml(
            TrainingReportComparison comparison) {
        return TrainingReportComparisonJUnitXml.render(comparison);
    }

    public static String trainingReportComparisonJUnitXml(
            TrainingReportComparisonExport export) {
        return TrainingReportComparisonJUnitXml.render(export);
    }

    public static String trainingReportComparisonArtifactMarkdown(
            TrainingReportComparisonArtifacts.ArtifactBundle bundle) {
        return TrainingReportComparisonMarkdown.render(bundle);
    }

    public static String trainingReportComparisonArtifactVerificationMarkdown(
            TrainingReportComparisonArtifacts.ArtifactVerification verification) {
        return TrainingReportComparisonMarkdown.render(verification);
    }

    public static String trainingReportComparisonArtifactVerificationJUnitXml(
            TrainingReportComparisonArtifacts.ArtifactVerification verification) {
        return TrainingReportComparisonJUnitXml.render(verification);
    }

    public static TrainingReportComparisonArtifacts.ArtifactBundle writeTrainingReportComparisonArtifacts(
            Path directory,
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportComparisonArtifacts.write(
                directory,
                compareTrainingReports(baselineReportFile, candidateReportFile));
    }

    public static TrainingReportComparisonArtifacts.ArtifactBundle writeTrainingReportComparisonArtifacts(
            Path directory,
            TrainingReportComparison comparison) throws IOException {
        return TrainingReportComparisonArtifacts.write(directory, comparison);
    }

    public static TrainingReportComparisonArtifacts.ArtifactBundle writeTrainingReportComparisonArtifacts(
            Path directory,
            TrainingReportComparisonExport export) throws IOException {
        return TrainingReportComparisonArtifacts.write(directory, export);
    }

    public static TrainingReportComparisonArtifacts.ArtifactBundle refreshTrainingReportComparisonArtifacts(
            Path directory) throws IOException {
        return TrainingReportComparisonArtifacts.refreshDerived(directory);
    }

    public static TrainingReportComparisonArtifacts.ArtifactInspection readTrainingReportComparisonArtifacts(
            Path directory) throws IOException {
        return TrainingReportComparisonArtifacts.read(directory);
    }

    public static TrainingReportComparisonArtifacts.ArtifactVerification verifyTrainingReportComparisonArtifacts(
            TrainingReportComparisonArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportComparisonArtifacts.verify(bundle);
    }

    public static TrainingReportComparisonArtifacts.ArtifactVerification verifyTrainingReportComparisonArtifacts(
            Path directory,
            String expectedJsonSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) throws IOException {
        return TrainingReportComparisonArtifacts.verify(
                directory,
                expectedJsonSha256,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256);
    }

    public static TrainingReportComparisonArtifacts.ArtifactVerification verifyTrainingReportComparisonArtifacts(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) throws IOException {
        return TrainingReportComparisonArtifacts.verify(
                directory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256);
    }

    public static TrainingReportComparisonArtifacts.ArtifactVerification verifyTrainingReportComparisonArtifacts(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) throws IOException {
        return TrainingReportComparisonArtifacts.verify(
                directory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256);
    }

    public static String trainingReportComparisonActionPlanSchema() {
        return TrainingReportAdvisor.comparisonActionPlanSchema();
    }

    public static Map<String, Object> trainingReportComparisonActionPlan(
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportAdvisor.comparisonActionPlan(
                TrainingReportReader.readCanonical(baselineReportFile),
                TrainingReportReader.readCanonical(candidateReportFile));
    }

    public static Map<String, Object> trainingReportComparisonActionPlan(
            TrainingReport baseline,
            TrainingReport candidate) {
        return TrainingReportAdvisor.comparisonActionPlan(baseline.payload(), candidate.payload());
    }

    public static TrainingReportComparisonActionPlanExport trainingReportComparisonActionPlanExport(
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportAdvisor.comparisonActionPlanExport(
                TrainingReportReader.readCanonical(baselineReportFile),
                TrainingReportReader.readCanonical(candidateReportFile));
    }

    public static TrainingReportComparisonActionPlanExport trainingReportComparisonActionPlanExport(
            TrainingReport baseline,
            TrainingReport candidate) {
        return TrainingReportAdvisor.comparisonActionPlanExport(baseline.payload(), candidate.payload());
    }

    public static TrainingReportComparisonActionPlanExport readTrainingReportComparisonActionPlanExport(
            Path jsonFile) throws IOException {
        return TrainingReportComparisonActionPlanExport.readJson(jsonFile);
    }

    public static String trainingReportComparisonActionPlanMarkdown(
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportAdvisor.comparisonActionPlanMarkdown(
                TrainingReportReader.readCanonical(baselineReportFile),
                TrainingReportReader.readCanonical(candidateReportFile));
    }

    public static String trainingReportComparisonActionPlanMarkdown(
            TrainingReport baseline,
            TrainingReport candidate) {
        return TrainingReportAdvisor.comparisonActionPlanMarkdown(baseline.payload(), candidate.payload());
    }

    public static String readTrainingReportComparisonActionPlanMarkdown(Path markdownFile) throws IOException {
        return TrainingReportComparisonActionPlanExport.readMarkdown(markdownFile);
    }

    public static TrainingReportComparisonActionPlanArtifacts.ArtifactBundle writeTrainingReportComparisonActionPlanArtifacts(
            Path directory,
            Path baselineReportFile,
            Path candidateReportFile) throws IOException {
        return TrainingReportComparisonActionPlanArtifacts.write(
                directory,
                trainingReportComparisonActionPlanExport(baselineReportFile, candidateReportFile));
    }

    public static TrainingReportComparisonActionPlanArtifacts.ArtifactBundle writeTrainingReportComparisonActionPlanArtifacts(
            Path directory,
            TrainingReport baseline,
            TrainingReport candidate) throws IOException {
        return TrainingReportComparisonActionPlanArtifacts.write(
                directory,
                trainingReportComparisonActionPlanExport(baseline, candidate));
    }

    public static TrainingReportComparisonActionPlanArtifacts.ArtifactBundle writeTrainingReportComparisonActionPlanArtifacts(
            Path directory,
            TrainingReportComparisonActionPlanExport export) throws IOException {
        return TrainingReportComparisonActionPlanArtifacts.write(directory, export);
    }

    public static TrainingReportComparisonActionPlanArtifacts.Verification verifyTrainingReportComparisonActionPlanArtifacts(
            TrainingReportComparisonActionPlanArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportComparisonActionPlanArtifacts.verify(bundle);
    }

    public static TrainingReportComparisonActionPlanArtifacts.Verification verifyTrainingReportComparisonActionPlanArtifacts(
            Path manifestFile) throws IOException {
        return TrainingReportComparisonActionPlanArtifacts.verify(manifestFile);
    }

    public static String trainingReportComparisonActionPlanArtifactVerificationMarkdown(
            TrainingReportComparisonActionPlanArtifacts.Verification verification) {
        return TrainingReportComparisonActionPlanArtifactVerificationMarkdown.render(verification);
    }

    public static TrainingReportPortfolio trainingReportPortfolio(Map<String, Path> reportFiles)
            throws IOException {
        return TrainingReportPortfolio.fromFiles(reportFiles);
    }

    public static TrainingReportPortfolio trainingReportPortfolioFromReports(Map<String, TrainingReport> reports) {
        return TrainingReportPortfolio.fromReports(reports);
    }

    public static TrainingReportPortfolioExport trainingReportPortfolioExport(Map<String, Path> reportFiles)
            throws IOException {
        return TrainingReportPortfolio.fromFiles(reportFiles).export();
    }

    public static TrainingReportPortfolioExport trainingReportPortfolioExport(
            Map<String, Path> reportFiles,
            String baselineName) throws IOException {
        return TrainingReportPortfolio.fromFiles(reportFiles).exportAgainst(baselineName);
    }

    public static TrainingReportPortfolioExport trainingReportPortfolioExport(TrainingReportPortfolio portfolio) {
        return portfolio.export();
    }

    public static TrainingReportPortfolioExport trainingReportPortfolioExport(
            TrainingReportPortfolio portfolio,
            String baselineName) {
        return portfolio.exportAgainst(baselineName);
    }

    public static String trainingReportPortfolioMarkdown(Map<String, Path> reportFiles)
            throws IOException {
        return TrainingReportPortfolioMarkdown.render(TrainingReportPortfolio.fromFiles(reportFiles).export());
    }

    public static String trainingReportPortfolioMarkdown(
            Map<String, Path> reportFiles,
            String baselineName) throws IOException {
        return TrainingReportPortfolioMarkdown.render(
                TrainingReportPortfolio.fromFiles(reportFiles).exportAgainst(baselineName));
    }

    public static String trainingReportPortfolioMarkdown(TrainingReportPortfolio portfolio) {
        return TrainingReportPortfolioMarkdown.render(portfolio);
    }

    public static String trainingReportPortfolioMarkdown(
            TrainingReportPortfolio portfolio,
            String baselineName) {
        return TrainingReportPortfolioMarkdown.render(portfolio, baselineName);
    }

    public static String trainingReportPortfolioMarkdown(TrainingReportPortfolioExport export) {
        return TrainingReportPortfolioMarkdown.render(export);
    }

    public static String trainingReportPortfolioArtifactMarkdown(
            TrainingReportPortfolioArtifacts.ArtifactBundle bundle) {
        return TrainingReportPortfolioMarkdown.render(bundle);
    }

    public static String trainingReportPortfolioArtifactVerificationMarkdown(
            TrainingReportPortfolioArtifacts.ArtifactVerification verification) {
        return TrainingReportPortfolioMarkdown.render(verification);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactBundle writeTrainingReportPortfolioExportArtifacts(
            Path directory,
            Map<String, Path> reportFiles) throws IOException {
        return TrainingReportPortfolioArtifacts.write(
                directory,
                TrainingReportPortfolio.fromFiles(reportFiles));
    }

    public static TrainingReportPortfolioArtifacts.ArtifactBundle writeTrainingReportPortfolioExportArtifacts(
            Path directory,
            Map<String, Path> reportFiles,
            String baselineName) throws IOException {
        return TrainingReportPortfolioArtifacts.write(
                directory,
                TrainingReportPortfolio.fromFiles(reportFiles),
                baselineName);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactBundle writeTrainingReportPortfolioExportArtifacts(
            Path directory,
            TrainingReportPortfolio portfolio) throws IOException {
        return TrainingReportPortfolioArtifacts.write(directory, portfolio);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactBundle writeTrainingReportPortfolioExportArtifacts(
            Path directory,
            TrainingReportPortfolio portfolio,
            String baselineName) throws IOException {
        return TrainingReportPortfolioArtifacts.write(directory, portfolio, baselineName);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactBundle writeTrainingReportPortfolioExportArtifacts(
            Path directory,
            TrainingReportPortfolioExport export) throws IOException {
        return TrainingReportPortfolioArtifacts.write(directory, export);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactBundle refreshTrainingReportPortfolioExportArtifacts(
            Path directory) throws IOException {
        return TrainingReportPortfolioArtifacts.refreshDerived(directory);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactInspection readTrainingReportPortfolioExportArtifacts(
            Path directory) throws IOException {
        return TrainingReportPortfolioArtifacts.read(directory);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestBundle writeTrainingReportPortfolioExportArtifactManifest(
            TrainingReportPortfolioArtifacts.ArtifactBundle artifacts) throws IOException {
        return TrainingReportPortfolioArtifactManifest.write(artifacts);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestInspection readTrainingReportPortfolioExportArtifactManifest(
            Path directory) throws IOException {
        return TrainingReportPortfolioArtifactManifest.read(directory);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verifyTrainingReportPortfolioExportArtifactManifest(
            TrainingReportPortfolioArtifactManifest.ManifestBundle manifest) throws IOException {
        return TrainingReportPortfolioArtifactManifest.verify(manifest);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verifyTrainingReportPortfolioExportArtifactManifest(
            Path directory) throws IOException {
        return TrainingReportPortfolioArtifactManifest.verify(directory);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verifyTrainingReportPortfolioExportArtifactManifest(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return TrainingReportPortfolioArtifactManifest.verify(directory, expectedManifestSha256);
    }

    public static TrainingReportPortfolioArtifactPackage.PackageBundle writeTrainingReportPortfolioExportArtifactPackage(
            Path directory,
            Map<String, Path> reportFiles) throws IOException {
        return TrainingReportPortfolioArtifactPackage.write(directory, reportFiles);
    }

    public static TrainingReportPortfolioArtifactPackage.PackageBundle writeTrainingReportPortfolioExportArtifactPackage(
            Path directory,
            Map<String, Path> reportFiles,
            String baselineName) throws IOException {
        return TrainingReportPortfolioArtifactPackage.write(directory, reportFiles, baselineName);
    }

    public static TrainingReportPortfolioArtifactPackage.PackageBundle writeTrainingReportPortfolioExportArtifactPackage(
            Path directory,
            TrainingReportPortfolioExport export) throws IOException {
        return TrainingReportPortfolioArtifactPackage.write(directory, export);
    }

    public static TrainingReportPortfolioArtifactPackage.VerifiedPackageBundle
            writeAndVerifyTrainingReportPortfolioExportArtifactPackage(
                    Path directory,
                    Map<String, Path> reportFiles) throws IOException {
        return TrainingReportPortfolioArtifactPackage.writeAndVerify(directory, reportFiles);
    }

    public static TrainingReportPortfolioArtifactPackage.VerifiedPackageBundle
            writeAndVerifyTrainingReportPortfolioExportArtifactPackage(
                    Path directory,
                    Map<String, Path> reportFiles,
                    String baselineName) throws IOException {
        return TrainingReportPortfolioArtifactPackage.writeAndVerify(directory, reportFiles, baselineName);
    }

    public static TrainingReportPortfolioArtifactPackage.VerifiedPackageBundle
            writeAndVerifyTrainingReportPortfolioExportArtifactPackage(
                    Path directory,
                    TrainingReportPortfolioExport export) throws IOException {
        return TrainingReportPortfolioArtifactPackage.writeAndVerify(directory, export);
    }

    public static TrainingReportPortfolioArtifactPackage.PackageInspection readTrainingReportPortfolioExportArtifactPackage(
            Path directory) throws IOException {
        return TrainingReportPortfolioArtifactPackage.read(directory);
    }

    public static TrainingReportPortfolioArtifactPackage.PackageRefresh refreshTrainingReportPortfolioExportArtifactPackage(
            Path directory) throws IOException {
        return TrainingReportPortfolioArtifactPackage.refreshComplete(directory);
    }

    public static TrainingReportPortfolioArtifactPackage.PackageRefresh refreshTrainingReportPortfolioExportArtifactPackage(
            Path directory,
            TrainingReportPortfolioArtifactPackage.Options options) throws IOException {
        return TrainingReportPortfolioArtifactPackage.refreshComplete(directory, options);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verifyTrainingReportPortfolioExportArtifactPackage(
            TrainingReportPortfolioArtifactPackage.PackageBundle bundle) throws IOException {
        return TrainingReportPortfolioArtifactPackage.verify(bundle);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verifyTrainingReportPortfolioExportArtifactPackage(
            Path directory) throws IOException {
        return TrainingReportPortfolioArtifactPackage.verify(directory);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verifyTrainingReportPortfolioExportArtifactPackage(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return TrainingReportPortfolioArtifactPackage.verify(directory, expectedManifestSha256);
    }

    public static String trainingReportPortfolioArtifactPackageVerificationJson(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        return TrainingReportPortfolioArtifactPackageReport.renderJson(verification);
    }

    public static String trainingReportPortfolioArtifactPackageContentFingerprint(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        return TrainingReportPortfolioArtifactPackageReport.contentFingerprint(verification);
    }

    public static String trainingReportPortfolioArtifactPackageVerificationMarkdown(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        return TrainingReportPortfolioArtifactPackageReport.renderMarkdown(verification);
    }

    public static String trainingReportPortfolioArtifactPackageVerificationJunitXml(
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        return TrainingReportPortfolioArtifactPackageReport.renderJunitXml(verification);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportBundle
            writeTrainingReportPortfolioArtifactPackageVerificationReport(
                    Path directory,
                    TrainingReportPortfolioArtifactManifest.ManifestVerification verification) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.write(directory, verification);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportBundle
            verifyTrainingReportPortfolioArtifactPackageAndWriteVerificationReport(
                    Path packageDirectory) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.verifyAndWrite(packageDirectory);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportBundle
            verifyTrainingReportPortfolioArtifactPackageAndWriteVerificationReport(
                    Path packageDirectory,
                    Path reportDirectory) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.verifyAndWrite(packageDirectory, reportDirectory);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportInspection
            readTrainingReportPortfolioArtifactPackageVerificationReport(
                    Path directory) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.read(directory);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportVerification
            verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                    TrainingReportPortfolioArtifactPackageReport.ReportBundle bundle) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.verify(bundle);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportVerification
            verifyTrainingReportPortfolioArtifactPackageVerificationReport(
                    Path directory,
                    String expectedJsonSha256,
                    String expectedMarkdownSha256,
                    String expectedJunitXmlSha256) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.verify(
                directory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency
            verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                    TrainingReportPortfolioArtifactPackageReport.ReportBundle reportBundle,
                    Path packageDirectory) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.verifyAgainstPackage(reportBundle, packageDirectory);
    }

    public static TrainingReportPortfolioArtifactPackageReport.ReportPackageConsistency
            verifyTrainingReportPortfolioArtifactPackageVerificationReportAgainstPackage(
                    Path reportDirectory,
                    Path packageDirectory,
                    String expectedJsonSha256,
                    String expectedMarkdownSha256,
                    String expectedJunitXmlSha256) throws IOException {
        return TrainingReportPortfolioArtifactPackageReport.verifyAgainstPackage(
                reportDirectory,
                packageDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactVerification verifyTrainingReportPortfolioExportArtifacts(
            TrainingReportPortfolioArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportPortfolioArtifacts.verify(bundle);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactVerification verifyTrainingReportPortfolioExportArtifacts(
            Path directory,
            String expectedJsonSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256) throws IOException {
        return TrainingReportPortfolioArtifacts.verify(
                directory,
                expectedJsonSha256,
                expectedLeaderboardCsvSha256,
                expectedComparisonMetricsCsvSha256,
                expectedComparisonFindingsCsvSha256);
    }

    public static TrainingReportPortfolioArtifacts.ArtifactVerification verifyTrainingReportPortfolioExportArtifacts(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256) throws IOException {
        return TrainingReportPortfolioArtifacts.verify(
                directory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedLeaderboardCsvSha256,
                expectedComparisonMetricsCsvSha256,
                expectedComparisonFindingsCsvSha256);
    }

    public static TrainingReportPortfolio.PromotionDecision trainingReportPromotionDecision(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity) throws IOException {
        return TrainingReportPortfolio.fromFiles(reportFiles)
                .promotionDecision(baselineName, maxAllowedDiagnosticSeverity);
    }

    public static TrainingReportPortfolio.PromotionDecision trainingReportPromotionDecision(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy) throws IOException {
        return TrainingReportPortfolio.fromFiles(reportFiles)
                .promotionDecision(baselineName, policy);
    }

    public static TrainingReportPortfolio.PromotionReview trainingReportPromotionReview(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity) throws IOException {
        return TrainingReportPortfolio.fromFiles(reportFiles)
                .promotionReview(baselineName, maxAllowedDiagnosticSeverity);
    }

    public static TrainingReportPortfolio.PromotionReview trainingReportPromotionReview(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy) throws IOException {
        return TrainingReportPortfolio.fromFiles(reportFiles)
                .promotionReview(baselineName, policy);
    }

    public static String trainingReportPromotionMarkdown(
            TrainingReportPortfolio.PromotionReview review) {
        return TrainingReportPromotionMarkdown.render(review);
    }

    public static String trainingReportPromotionMarkdown(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity) throws IOException {
        return TrainingReportPromotionMarkdown.render(
                trainingReportPromotionReview(reportFiles, baselineName, maxAllowedDiagnosticSeverity));
    }

    public static String trainingReportPromotionMarkdown(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy) throws IOException {
        return TrainingReportPromotionMarkdown.render(
                trainingReportPromotionReview(reportFiles, baselineName, policy));
    }

    public static TrainingReportPromotionArtifacts.ArtifactBundle writeTrainingReportPromotionArtifacts(
            Path outputDirectory,
            TrainingReportPortfolio.PromotionReview review) throws IOException {
        return TrainingReportPromotionArtifacts.write(outputDirectory, review);
    }

    public static TrainingReportPromotionArtifacts.ArtifactBundle writeTrainingReportPromotionArtifacts(
            Path outputDirectory,
            TrainingReportPortfolio.PromotionReview review,
            TrainingReportPromotionArtifacts.Options options) throws IOException {
        return TrainingReportPromotionArtifacts.write(outputDirectory, review, options);
    }

    public static TrainingReportPromotionArtifacts.ArtifactBundle writeTrainingReportPromotionArtifacts(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionArtifacts.write(
                outputDirectory,
                trainingReportPromotionReview(reportFiles, baselineName, maxAllowedDiagnosticSeverity));
    }

    public static TrainingReportPromotionArtifacts.ArtifactBundle writeTrainingReportPromotionArtifacts(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionArtifacts.write(
                outputDirectory,
                trainingReportPromotionReview(reportFiles, baselineName, policy));
    }

    public static TrainingReportPromotionArtifacts.ArtifactInspection readTrainingReportPromotionArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionArtifacts.read(outputDirectory);
    }

    public static TrainingReportPromotionArtifacts.ArtifactInspection readTrainingReportPromotionArtifacts(
            Path outputDirectory,
            TrainingReportPromotionArtifacts.Options options) throws IOException {
        return TrainingReportPromotionArtifacts.read(outputDirectory, options);
    }

    public static TrainingReportPromotionArtifacts.ArtifactVerification verifyTrainingReportPromotionArtifacts(
            TrainingReportPromotionArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportPromotionArtifacts.verify(bundle);
    }

    public static TrainingReportPromotionArtifacts.ArtifactVerification verifyTrainingReportPromotionArtifacts(
            Path outputDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256) throws IOException {
        return TrainingReportPromotionArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256);
    }

    public static TrainingReportPromotionArtifacts.ArtifactVerification verifyTrainingReportPromotionArtifacts(
            Path outputDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            TrainingReportPromotionArtifacts.Options options) throws IOException {
        return TrainingReportPromotionArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                options);
    }

    public static List<TrainingReportPromotionArtifacts.SourceReport> trainingReportPromotionSourceReports(
            TrainingReportPromotionArtifacts.ArtifactInspection inspection) {
        return TrainingReportPromotionArtifacts.sourceReports(inspection);
    }

    public static TrainingReportPromotionArtifacts.SourceVerification verifyTrainingReportPromotionSourceReports(
            TrainingReportPromotionArtifacts.ArtifactInspection inspection) throws IOException {
        return TrainingReportPromotionArtifacts.verifySourceReports(inspection);
    }

    public static TrainingReportPromotionArtifacts.SourceVerification verifyTrainingReportPromotionSourceReports(
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionArtifacts.verifySourceReports(outputDirectory);
    }

    public static TrainingReportPromotionArtifacts.SourceVerification verifyTrainingReportPromotionSourceReports(
            Path outputDirectory,
            TrainingReportPromotionArtifacts.Options options) throws IOException {
        return TrainingReportPromotionArtifacts.verifySourceReports(outputDirectory, options);
    }

    public static TrainingReportPromotionGate.Result runTrainingReportPromotionGate(
            TrainingReportPromotionGate.Request request) throws IOException {
        return TrainingReportPromotionGate.evaluate(request);
    }

    public static TrainingReportPromotionGate.Result runTrainingReportPromotionGate(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportPortfolio.PromotionPolicy policy,
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionGate.evaluate(
                reportFiles,
                baselineName,
                policy,
                outputDirectory);
    }

    public static TrainingReportPromotionGate.Result runTrainingReportPromotionGate(
            Map<String, Path> reportFiles,
            String baselineName,
            TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionGate.evaluate(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory);
    }

    public static String trainingReportPromotionGateMarkdown(
            TrainingReportPromotionGate.Result result) {
        return TrainingReportPromotionGateMarkdown.render(result);
    }

    public static String trainingReportPromotionGateJUnitXml(
            TrainingReportPromotionGate.Result result) {
        return TrainingReportPromotionGateJUnitXml.render(result);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactBundle writeTrainingReportPromotionGateArtifacts(
            Path outputDirectory,
            TrainingReportPromotionGate.Result result) throws IOException {
        return TrainingReportPromotionGateArtifacts.write(outputDirectory, result);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactBundle writeTrainingReportPromotionGateArtifacts(
            Path outputDirectory,
            TrainingReportPromotionGate.Result result,
            TrainingReportPromotionGateArtifacts.Options options) throws IOException {
        return TrainingReportPromotionGateArtifacts.write(outputDirectory, result, options);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactInspection readTrainingReportPromotionGateArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifacts.read(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactInspection readTrainingReportPromotionGateArtifacts(
            Path outputDirectory,
            TrainingReportPromotionGateArtifacts.Options options) throws IOException {
        return TrainingReportPromotionGateArtifacts.read(outputDirectory, options);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactInspection refreshTrainingReportPromotionGateArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifacts.refreshDerived(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactInspection refreshTrainingReportPromotionGateArtifacts(
            Path outputDirectory,
            TrainingReportPromotionGateArtifacts.Options options) throws IOException {
        return TrainingReportPromotionGateArtifacts.refreshDerived(outputDirectory, options);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactVerification verifyTrainingReportPromotionGateArtifacts(
            TrainingReportPromotionGateArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportPromotionGateArtifacts.verify(bundle);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactVerification verifyTrainingReportPromotionGateArtifacts(
            Path outputDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) throws IOException {
        return TrainingReportPromotionGateArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactVerification verifyTrainingReportPromotionGateArtifacts(
            Path outputDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            TrainingReportPromotionGateArtifacts.Options options) throws IOException {
        return TrainingReportPromotionGateArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256,
                options);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestBundle
            writeTrainingReportPromotionGateArtifactManifest(
                    TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.write(artifacts);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestBundle
            writeTrainingReportPromotionGateArtifactManifest(
                    TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
                    Instant generatedAt) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.write(artifacts, generatedAt);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestBundle
            writeTrainingReportPromotionGateArtifactManifest(
                    TrainingReportPromotionGateArtifacts.ArtifactBundle artifacts,
                    TrainingReportPromotionGateArtifactManifest.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.write(artifacts, options);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestInspection
            readTrainingReportPromotionGateArtifactManifest(Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.read(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestInspection
            readTrainingReportPromotionGateArtifactManifest(
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactManifest.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.read(outputDirectory, options);
    }

    public static TrainingReportPromotionGateArtifacts.ArtifactInspection
            readTrainingReportPromotionGateArtifacts(
                    TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.readArtifacts(manifest);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactManifest(
                    TrainingReportPromotionGateArtifactManifest.ManifestBundle bundle) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.verify(bundle);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactManifest(Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.verify(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactManifest(
                    Path outputDirectory,
                    String expectedManifestSha256) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.verify(outputDirectory, expectedManifestSha256);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactManifest(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactManifest.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactManifest.verify(outputDirectory, expectedManifestSha256, options);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageBundle
            writeTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGate.Result result) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.write(outputDirectory, result);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageBundle
            writeTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGate.Result result,
                    Instant generatedAt) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.write(outputDirectory, result, generatedAt);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageBundle
            writeTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGate.Result result,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.write(outputDirectory, result, options);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageBundle
            runTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportPortfolio.PromotionPolicy policy,
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.run(
                reportFiles,
                baselineName,
                policy,
                outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageBundle
            runTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportPortfolio.PromotionPolicy policy,
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Instant generatedAt) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.run(
                reportFiles,
                baselineName,
                policy,
                outputDirectory,
                options,
                generatedAt);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageBundle
            runTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.runWithSeverity(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageBundle
            runTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Instant generatedAt) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.runWithSeverity(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                options,
                generatedAt);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle
            writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGate.Result result) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeAndVerify(outputDirectory, result);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle
            writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGate.Result result,
                    Instant generatedAt) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeAndVerify(outputDirectory, result, generatedAt);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle
            writeAndVerifyTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGate.Result result,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Instant generatedAt,
                    Path reportDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeAndVerify(
                outputDirectory,
                result,
                options,
                generatedAt,
                reportDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle
            runAndVerifyTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportPortfolio.PromotionPolicy policy,
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.runAndVerify(
                reportFiles,
                baselineName,
                policy,
                outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle
            runAndVerifyTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportPortfolio.PromotionPolicy policy,
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Instant generatedAt,
                    Path reportDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.runAndVerify(
                reportFiles,
                baselineName,
                policy,
                outputDirectory,
                options,
                generatedAt,
                reportDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle
            runAndVerifyTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.runWithSeverityAndVerify(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle
            runAndVerifyTrainingReportPromotionGateArtifactPackage(
                    Map<String, Path> reportFiles,
                    String baselineName,
                    TrainingReportDiagnostics.Severity maxAllowedDiagnosticSeverity,
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Instant generatedAt,
                    Path reportDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.runWithSeverityAndVerify(
                reportFiles,
                baselineName,
                maxAllowedDiagnosticSeverity,
                outputDirectory,
                options,
                generatedAt,
                reportDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageInspection
            readTrainingReportPromotionGateArtifactPackage(Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.read(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageInspection
            readTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.read(outputDirectory, options);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageRefresh
            refreshTrainingReportPromotionGateArtifactPackage(Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.refreshComplete(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageRefresh
            refreshTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.refreshComplete(outputDirectory, options);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactPackage(
                    TrainingReportPromotionGateArtifactPackage.PackageBundle bundle) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verify(bundle);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactPackage(Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verify(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    String expectedManifestSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verify(outputDirectory, expectedManifestSha256);
    }

    public static TrainingReportPromotionGateArtifactManifest.ManifestVerification
            verifyTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verify(outputDirectory, expectedManifestSha256, options);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageVerification
            verifyCompleteTrainingReportPromotionGateArtifactPackage(
                    TrainingReportPromotionGateArtifactPackage.PackageBundle bundle) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyComplete(bundle);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageVerification
            verifyCompleteTrainingReportPromotionGateArtifactPackage(Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyComplete(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageVerification
            verifyCompleteTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    String expectedManifestSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyComplete(outputDirectory, expectedManifestSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.PackageVerification
            verifyCompleteTrainingReportPromotionGateArtifactPackage(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyComplete(
                outputDirectory,
                expectedManifestSha256,
                options);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReport(
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReport(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReport(
                    Path outputDirectory,
                    String expectedManifestSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReport(
                outputDirectory,
                expectedManifestSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReport(
                    Path outputDirectory,
                    Path reportFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReport(outputDirectory, reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReport(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReport(
                outputDirectory,
                expectedManifestSha256,
                options);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReport(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Path reportFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReport(
                outputDirectory,
                expectedManifestSha256,
                options,
                reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReport
            writeTrainingReportPromotionGatePackageVerificationReport(
                    Path reportFile,
                    TrainingReportPromotionGateArtifactPackage.PackageVerification verification) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationReport(reportFile, verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportInspection
            readTrainingReportPromotionGatePackageVerificationReport(Path reportFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationReport(reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportVerification
            verifyTrainingReportPromotionGatePackageVerificationReport(Path reportFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReport(reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportVerification
            verifyTrainingReportPromotionGatePackageVerificationReport(
                    Path reportFile,
                    String expectedReportSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReport(reportFile, expectedReportSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationMarkdownReport(
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteMarkdownReport(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationMarkdownReport(
                    Path outputDirectory,
                    String expectedManifestSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteMarkdownReport(
                outputDirectory,
                expectedManifestSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationMarkdownReport(
                    Path outputDirectory,
                    Path markdownFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteMarkdownReport(
                outputDirectory,
                markdownFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationMarkdownReport(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteMarkdownReport(
                outputDirectory,
                expectedManifestSha256,
                options);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationMarkdownReport(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Path markdownFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteMarkdownReport(
                outputDirectory,
                expectedManifestSha256,
                options,
                markdownFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationMarkdownReport
            writeTrainingReportPromotionGatePackageVerificationMarkdownReport(
                    Path markdownFile,
                    TrainingReportPromotionGateArtifactPackage.PackageVerification verification) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationMarkdownReport(markdownFile, verification);
    }

    public static String renderTrainingReportPromotionGatePackageVerificationMarkdown(
            TrainingReportPromotionGateArtifactPackage.PackageVerification verification) {
        return TrainingReportPromotionGateArtifactPackage.renderVerificationMarkdown(verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteJUnitXmlReport(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                    Path outputDirectory,
                    String expectedManifestSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteJUnitXmlReport(
                outputDirectory,
                expectedManifestSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                    Path outputDirectory,
                    Path junitXmlFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteJUnitXmlReport(
                outputDirectory,
                junitXmlFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteJUnitXmlReport(
                outputDirectory,
                expectedManifestSha256,
                options);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport
            verifyAndWriteTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Path junitXmlFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteJUnitXmlReport(
                outputDirectory,
                expectedManifestSha256,
                options,
                junitXmlFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationJUnitXmlReport
            writeTrainingReportPromotionGatePackageVerificationJUnitXmlReport(
                    Path junitXmlFile,
                    TrainingReportPromotionGateArtifactPackage.PackageVerification verification) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationJUnitXmlReport(junitXmlFile, verification);
    }

    public static String renderTrainingReportPromotionGatePackageVerificationJUnitXml(
            TrainingReportPromotionGateArtifactPackage.PackageVerification verification) {
        return TrainingReportPromotionGateArtifactPackage.renderVerificationJUnitXml(verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundle
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReports(
                    Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReports(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundle
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReports(
                    Path outputDirectory,
                    String expectedManifestSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReports(
                outputDirectory,
                expectedManifestSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundle
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReports(
                    Path outputDirectory,
                    Path reportDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReports(
                outputDirectory,
                reportDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundle
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReports(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReports(
                outputDirectory,
                expectedManifestSha256,
                options);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundle
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReports(
                    Path outputDirectory,
                    String expectedManifestSha256,
                    TrainingReportPromotionGateArtifactPackage.Options options,
                    Path reportDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyCompleteAndWriteReports(
                outputDirectory,
                expectedManifestSha256,
                options,
                reportDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundle
            writeTrainingReportPromotionGatePackageVerificationReports(
                    Path reportDirectory,
                    TrainingReportPromotionGateArtifactPackage.PackageVerification verification) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationReports(reportDirectory, verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection
            readTrainingReportPromotionGatePackageVerificationReports(Path reportDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationReportBundle(reportDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification
            verifyTrainingReportPromotionGatePackageVerificationReports(Path reportDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundle(reportDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification
            verifyTrainingReportPromotionGatePackageVerificationReports(
                    Path reportDirectory,
                    String expectedJsonReportSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundle(
                reportDirectory,
                expectedJsonReportSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceipt
            writeTrainingReportPromotionGatePackageVerificationReportsReceipt(
                    Path receiptFile,
                    TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification verification)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationReportBundleReceipt(
                receiptFile,
                verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceipt
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReportsReceipt(
                    Path reportDirectory,
                    Path receiptFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundleAndWriteReceipt(
                reportDirectory,
                receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceipt
            verifyAndWriteTrainingReportPromotionGatePackageVerificationReportsReceipt(
                    Path reportDirectory,
                    String expectedJsonReportSha256,
                    Path receiptFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundleAndWriteReceipt(
                reportDirectory,
                expectedJsonReportSha256,
                receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptInspection
            readTrainingReportPromotionGatePackageVerificationReportsReceipt(Path receiptFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationReportBundleReceipt(receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification
            verifyTrainingReportPromotionGatePackageVerificationReportsReceipt(Path receiptFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundleReceipt(receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleReceiptVerification
            verifyTrainingReportPromotionGatePackageVerificationReportsReceipt(
                    Path receiptFile,
                    String expectedReceiptSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationReportBundleReceipt(
                receiptFile,
                expectedReceiptSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndex
            writeTrainingReportPromotionGatePackageVerificationIndex(
                    Path indexFile,
                    TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verifiedPackage)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationIndex(indexFile, verifiedPackage);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexInspection
            readTrainingReportPromotionGatePackageVerificationIndex(Path indexFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationIndex(indexFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification
            verifyTrainingReportPromotionGatePackageVerificationIndex(Path indexFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndex(indexFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification
            verifyTrainingReportPromotionGatePackageVerificationIndex(
                    Path indexFile,
                    String expectedIndexSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndex(indexFile, expectedIndexSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit
            auditTrainingReportPromotionGatePackageVerificationIndex(Path indexFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackage(indexFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit
            auditTrainingReportPromotionGatePackageVerificationIndex(
                    Path indexFile,
                    String expectedIndexSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackage(
                indexFile,
                expectedIndexSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit
            auditTrainingReportPromotionGatePackageVerificationIndex(
                    TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification verification)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackage(verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReport
            writeTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                    Path reportFile,
                    TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAudit audit)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationIndexPackageAuditReport(reportFile, audit);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportInspection
            readTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(Path reportFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationIndexPackageAuditReport(reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification
            verifyTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(Path reportFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndexPackageAuditReport(reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReportVerification
            verifyTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                    Path reportFile,
                    String expectedReportSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndexPackageAuditReport(
                reportFile,
                expectedReportSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReport
            auditAndWriteTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                    Path indexFile,
                    Path reportFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackageAndWriteReport(
                indexFile,
                reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexPackageAuditReport
            auditAndWriteTrainingReportPromotionGatePackageVerificationIndexPackageAuditReport(
                    Path indexFile,
                    String expectedIndexSha256,
                    Path reportFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.auditVerificationIndexPackageAndWriteReport(
                indexFile,
                expectedIndexSha256,
                reportFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceManifest
            writeTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                    Path evidenceFile,
                    TrainingReportPromotionGateArtifactPackage.VerifiedPackageBundle verifiedPackage)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationEvidenceManifest(
                evidenceFile,
                verifiedPackage);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceInspection
            readTrainingReportPromotionGatePackageVerificationEvidenceManifest(Path evidenceFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationEvidenceManifest(evidenceFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification
            verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(Path evidenceFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceManifest(evidenceFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification
            verifyTrainingReportPromotionGatePackageVerificationEvidenceManifest(
                    Path evidenceFile,
                    String expectedEvidenceSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceManifest(
                evidenceFile,
                expectedEvidenceSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceipt
            writeTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                    Path receiptFile,
                    TrainingReportPromotionGateArtifactPackage.VerificationEvidenceVerification verification)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationEvidenceReceipt(
                receiptFile,
                verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceipt
            verifyAndWriteTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                    Path evidenceFile,
                    Path receiptFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceManifestAndWriteReceipt(
                evidenceFile,
                receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceipt
            verifyAndWriteTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                    Path evidenceFile,
                    String expectedEvidenceSha256,
                    Path receiptFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceManifestAndWriteReceipt(
                evidenceFile,
                expectedEvidenceSha256,
                receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptInspection
            readTrainingReportPromotionGatePackageVerificationEvidenceReceipt(Path receiptFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationEvidenceReceipt(receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification
            verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(Path receiptFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceReceipt(receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationEvidenceReceiptVerification
            verifyTrainingReportPromotionGatePackageVerificationEvidenceReceipt(
                    Path receiptFile,
                    String expectedReceiptSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationEvidenceReceipt(
                receiptFile,
                expectedReceiptSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceipt
            writeTrainingReportPromotionGatePackageVerificationIndexReceipt(
                    Path receiptFile,
                    TrainingReportPromotionGateArtifactPackage.VerificationIndexVerification verification)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.writeVerificationIndexReceipt(receiptFile, verification);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptInspection
            readTrainingReportPromotionGatePackageVerificationIndexReceipt(Path receiptFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.readVerificationIndexReceipt(receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification
            verifyTrainingReportPromotionGatePackageVerificationIndexReceipt(Path receiptFile)
                    throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndexReceipt(receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceiptVerification
            verifyTrainingReportPromotionGatePackageVerificationIndexReceipt(
                    Path receiptFile,
                    String expectedReceiptSha256) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndexReceipt(
                receiptFile,
                expectedReceiptSha256);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceipt
            verifyAndWriteTrainingReportPromotionGatePackageVerificationIndexReceipt(
                    Path indexFile,
                    Path receiptFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndexAndWriteReceipt(
                indexFile,
                receiptFile);
    }

    public static TrainingReportPromotionGateArtifactPackage.VerificationIndexReceipt
            verifyAndWriteTrainingReportPromotionGatePackageVerificationIndexReceipt(
                    Path indexFile,
                    String expectedIndexSha256,
                    Path receiptFile) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifyVerificationIndexAndWriteReceipt(
                indexFile,
                expectedIndexSha256,
                receiptFile);
    }

    public static List<TrainingReportPromotionGateArtifactPackage.SourceReportSnapshot>
            trainingReportPromotionGatePackageSourceSnapshots(
                    TrainingReportPromotionGateArtifactPackage.PackageInspection inspection) {
        return TrainingReportPromotionGateArtifactPackage.sourceReportSnapshots(inspection);
    }

    public static List<String> expectedTrainingReportPromotionGatePackageSourceArtifacts(
            TrainingReportPromotionGateArtifactPackage.PackageInspection inspection) {
        return TrainingReportPromotionGateArtifactPackage.expectedSourceReportArtifactNames(inspection);
    }

    public static List<String> presentTrainingReportPromotionGatePackageSourceArtifacts(
            TrainingReportPromotionGateArtifactPackage.PackageInspection inspection) {
        return TrainingReportPromotionGateArtifactPackage.presentSourceReportArtifactNames(inspection);
    }

    public static List<String> missingTrainingReportPromotionGatePackageSourceArtifacts(
            TrainingReportPromotionGateArtifactPackage.PackageInspection inspection) {
        return TrainingReportPromotionGateArtifactPackage.missingSourceReportArtifactNames(inspection);
    }

    public static List<String> unexpectedTrainingReportPromotionGatePackageSourceArtifacts(
            TrainingReportPromotionGateArtifactPackage.PackageInspection inspection) {
        return TrainingReportPromotionGateArtifactPackage.unexpectedSourceReportArtifactNames(inspection);
    }

    public static TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification
            verifyTrainingReportPromotionGatePackageSourceSnapshots(
                    TrainingReportPromotionGateArtifactPackage.PackageInspection inspection) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifySourceReportSnapshots(inspection);
    }

    public static TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification
            verifyTrainingReportPromotionGatePackageSourceSnapshots(Path outputDirectory) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifySourceReportSnapshots(outputDirectory);
    }

    public static TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification
            verifyTrainingReportPromotionGatePackageSourceSnapshots(
                    Path outputDirectory,
                    TrainingReportPromotionGateArtifactPackage.Options options) throws IOException {
        return TrainingReportPromotionGateArtifactPackage.verifySourceReportSnapshots(outputDirectory, options);
    }

    public static List<Map<String, Object>> trainingReportHistory(Path reportFile) throws IOException {
        return TrainingReportReader.history(readTrainingReport(reportFile));
    }

    public static List<TrainingReportEpochSnapshot> trainingReportEpochSnapshots(Path reportFile) throws IOException {
        return TrainingReportReader.epochSnapshots(readTrainingReport(reportFile));
    }

    public static List<TrainingReportEpochSnapshot> trainingReportEpochSnapshots(TrainingReport report) {
        return report.epochSnapshots();
    }

    public static Optional<TrainingReportEpochSnapshot> trainingReportLatestEpochSnapshot(Path reportFile)
            throws IOException {
        return TrainingReportReader.latestEpochSnapshot(readTrainingReport(reportFile));
    }

    public static Optional<TrainingReportEpochSnapshot> trainingReportLatestEpochSnapshot(TrainingReport report) {
        return report.latestEpochSnapshot();
    }

    public static Optional<TrainingReportEpochSnapshot> trainingReportEpochSnapshot(Path reportFile, int epoch)
            throws IOException {
        return TrainingReportReader.epochSnapshot(readTrainingReport(reportFile), epoch);
    }

    public static Optional<TrainingReportEpochSnapshot> trainingReportEpochSnapshot(TrainingReport report, int epoch) {
        return report.epochSnapshot(epoch);
    }

    public static TrainingReportSeries trainingReportTrainLossSeries(Path reportFile) throws IOException {
        return TrainingReportReader.trainLossSeries(readTrainingReport(reportFile));
    }

    public static TrainingReportSeries trainingReportTrainLossSeries(TrainingReport report) {
        return report.trainLossSeries();
    }

    public static TrainingReportSeries trainingReportValidationLossSeries(Path reportFile) throws IOException {
        return TrainingReportReader.validationLossSeries(readTrainingReport(reportFile));
    }

    public static TrainingReportSeries trainingReportValidationLossSeries(TrainingReport report) {
        return report.validationLossSeries();
    }

    public static TrainingReportSeries trainingReportLearningRateSeries(Path reportFile) throws IOException {
        return TrainingReportReader.learningRateSeries(readTrainingReport(reportFile));
    }

    public static TrainingReportSeries trainingReportLearningRateSeries(TrainingReport report) {
        return report.learningRateSeries();
    }

    public static TrainingReportSeries trainingReportGeneralizationGapSeries(Path reportFile) throws IOException {
        return TrainingReportReader.generalizationGapSeries(readTrainingReport(reportFile));
    }

    public static TrainingReportSeries trainingReportGeneralizationGapSeries(TrainingReport report) {
        return report.generalizationGapSeries();
    }

    public static TrainingReportSeries trainingReportValidationToTrainLossRatioSeries(Path reportFile)
            throws IOException {
        return TrainingReportReader.validationToTrainLossRatioSeries(readTrainingReport(reportFile));
    }

    public static TrainingReportSeries trainingReportValidationToTrainLossRatioSeries(TrainingReport report) {
        return report.validationToTrainLossRatioSeries();
    }

    public static TrainingReportSeries trainingReportTrainMetricSeries(Path reportFile, String metricName)
            throws IOException {
        return TrainingReportReader.trainMetricSeries(readTrainingReport(reportFile), metricName);
    }

    public static TrainingReportSeries trainingReportTrainMetricSeries(TrainingReport report, String metricName) {
        return report.trainMetricSeries(metricName);
    }

    public static TrainingReportSeries trainingReportValidationMetricSeries(Path reportFile, String metricName)
            throws IOException {
        return TrainingReportReader.validationMetricSeries(readTrainingReport(reportFile), metricName);
    }

    public static TrainingReportSeries trainingReportValidationMetricSeries(TrainingReport report, String metricName) {
        return report.validationMetricSeries(metricName);
    }

    public static TrainingReportSeries trainingReportGradientL2NormSeries(Path reportFile) throws IOException {
        return TrainingReportReader.gradientL2NormSeries(readTrainingReport(reportFile));
    }

    public static TrainingReportSeries trainingReportGradientL2NormSeries(TrainingReport report) {
        return report.gradientL2NormSeries();
    }

    public static TrainingReportSeries trainingReportParameterUpdateToParameterL2RatioSeries(Path reportFile)
            throws IOException {
        return TrainingReportReader.parameterUpdateToParameterL2RatioSeries(readTrainingReport(reportFile));
    }

    public static TrainingReportSeries trainingReportParameterUpdateToParameterL2RatioSeries(TrainingReport report) {
        return report.parameterUpdateToParameterL2RatioSeries();
    }

    public static TrainingReportSeriesBundle trainingReportSeriesBundle(Path reportFile) throws IOException {
        return TrainingReportReader.seriesBundle(readTrainingReport(reportFile));
    }

    public static TrainingReportSeriesBundle trainingReportSeriesBundle(TrainingReport report) {
        return report.seriesBundle();
    }

    public static TrainingReportSeriesExport trainingReportSeriesExport(Path reportFile) throws IOException {
        return TrainingReportReader.seriesExport(readTrainingReport(reportFile));
    }

    public static TrainingReportSeriesExport trainingReportSeriesExport(TrainingReport report) {
        return report.seriesExport();
    }

    public static Map<String, Object> trainingReportHistorySummary(Path reportFile) throws IOException {
        return TrainingReportReader.historySummary(readTrainingReport(reportFile));
    }

    public static TrainingReportHistoryOverview trainingReportHistoryOverview(Path reportFile) throws IOException {
        return TrainingReportReader.historyOverview(readTrainingReport(reportFile));
    }

    public static TrainingReportHistoryOverview trainingReportHistoryOverview(TrainingReport report) {
        return report.historyOverview();
    }

    public static List<Map<String, Object>> trainingReportDiagnostics(Path reportFile) throws IOException {
        return TrainingReportReader.diagnostics(readTrainingReport(reportFile));
    }

    public static List<TrainingReportDiagnostics.Finding> trainingReportDiagnosticFindings(Path reportFile)
            throws IOException {
        return TrainingReportReader.diagnosticFindings(readTrainingReport(reportFile));
    }

    public static Map<String, Object> trainingReportDiagnosticsSummary(Path reportFile) throws IOException {
        return TrainingReportReader.diagnosticsSummary(readTrainingReport(reportFile));
    }

    public static TrainingReportDiagnostics.Summary trainingReportDiagnosticSummary(Path reportFile)
            throws IOException {
        return TrainingReportReader.diagnosticSummary(readTrainingReport(reportFile));
    }

    public static Map<String, Object> trainingReportDiagnosticsProvenance(Path reportFile) throws IOException {
        return TrainingReportReader.diagnosticsProvenance(readTrainingReport(reportFile));
    }

    public static TrainingReportDiagnostics.Provenance trainingReportDiagnosticProvenance(Path reportFile)
            throws IOException {
        return TrainingReportReader.diagnosticProvenance(readTrainingReport(reportFile));
    }

    public static TrainingReportDiagnostics.GateResult trainingReportDiagnosticGate(
            Path reportFile,
            TrainingReportDiagnostics.Severity maxAllowedSeverity) throws IOException {
        return TrainingReportDiagnostics.gateFindings(
                TrainingReportReader.diagnosticFindings(readTrainingReport(reportFile)),
                maxAllowedSeverity);
    }

    public static TrainingReportValidationPolicy.Result validateTrainingReport(Path reportFile)
            throws IOException {
        return validateTrainingReport(reportFile, TrainingReportValidationPolicy.defaultPolicy());
    }

    public static TrainingReportValidationPolicy.Result validateTrainingReport(
            Path reportFile,
            TrainingReportValidationPolicy policy) throws IOException {
        return TrainingReportReader.readReport(reportFile).validate(policy);
    }

    public static TrainingReportValidationPolicy.Result validateTrainingReport(TrainingReport report) {
        return validateTrainingReport(report, TrainingReportValidationPolicy.defaultPolicy());
    }

    public static TrainingReportValidationPolicy.Result validateTrainingReport(
            TrainingReport report,
            TrainingReportValidationPolicy policy) {
        return report.validate(policy);
    }

    public static String trainingReportValidationMarkdown(Path reportFile) throws IOException {
        return trainingReportValidationMarkdown(validateTrainingReport(reportFile));
    }

    public static String trainingReportValidationMarkdown(
            Path reportFile,
            TrainingReportValidationPolicy policy) throws IOException {
        return trainingReportValidationMarkdown(validateTrainingReport(reportFile, policy));
    }

    public static String trainingReportValidationMarkdown(TrainingReport report) {
        return trainingReportValidationMarkdown(validateTrainingReport(report));
    }

    public static String trainingReportValidationMarkdown(TrainingReportValidationPolicy.Result result) {
        return TrainingReportValidationMarkdown.render(result);
    }

    public static String trainingReportValidationJUnitXml(Path reportFile) throws IOException {
        return trainingReportValidationJUnitXml(validateTrainingReport(reportFile));
    }

    public static String trainingReportValidationJUnitXml(
            Path reportFile,
            TrainingReportValidationPolicy policy) throws IOException {
        return trainingReportValidationJUnitXml(validateTrainingReport(reportFile, policy));
    }

    public static String trainingReportValidationJUnitXml(TrainingReport report) {
        return trainingReportValidationJUnitXml(validateTrainingReport(report));
    }

    public static String trainingReportValidationJUnitXml(TrainingReportValidationPolicy.Result result) {
        return TrainingReportValidationJUnitXml.render(result);
    }

    public static TrainingReportValidationArtifacts.ArtifactBundle writeTrainingReportValidationArtifacts(
            Path outputDirectory,
            Path reportFile) throws IOException {
        return writeTrainingReportValidationArtifacts(
                outputDirectory,
                validateTrainingReport(reportFile));
    }

    public static TrainingReportValidationArtifacts.ArtifactBundle writeTrainingReportValidationArtifacts(
            Path outputDirectory,
            Path reportFile,
            TrainingReportValidationPolicy policy) throws IOException {
        return writeTrainingReportValidationArtifacts(
                outputDirectory,
                validateTrainingReport(reportFile, policy));
    }

    public static TrainingReportValidationArtifacts.ArtifactBundle writeTrainingReportValidationArtifacts(
            Path outputDirectory,
            TrainingReport report) throws IOException {
        return writeTrainingReportValidationArtifacts(
                outputDirectory,
                validateTrainingReport(report));
    }

    public static TrainingReportValidationArtifacts.ArtifactBundle writeTrainingReportValidationArtifacts(
            Path outputDirectory,
            TrainingReportValidationPolicy.Result result) throws IOException {
        return TrainingReportValidationArtifacts.write(outputDirectory, result);
    }

    public static TrainingReportValidationArtifacts.ArtifactBundle writeTrainingReportValidationArtifacts(
            Path outputDirectory,
            TrainingReportValidationPolicy.Result result,
            TrainingReportValidationArtifacts.Options options) throws IOException {
        return TrainingReportValidationArtifacts.write(outputDirectory, result, options);
    }

    public static TrainingReportValidationArtifacts.ArtifactBundle refreshTrainingReportValidationArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportValidationArtifacts.refreshDerived(outputDirectory);
    }

    public static TrainingReportValidationArtifacts.ArtifactInspection readTrainingReportValidationArtifacts(
            Path outputDirectory) throws IOException {
        return TrainingReportValidationArtifacts.read(outputDirectory);
    }

    public static TrainingReportValidationArtifacts.ArtifactInspection readTrainingReportValidationArtifacts(
            Path outputDirectory,
            TrainingReportValidationArtifacts.Options options) throws IOException {
        return TrainingReportValidationArtifacts.read(outputDirectory, options);
    }

    public static TrainingReportValidationArtifacts.ArtifactVerification verifyTrainingReportValidationArtifacts(
            TrainingReportValidationArtifacts.ArtifactBundle bundle) throws IOException {
        return TrainingReportValidationArtifacts.verify(bundle);
    }

    public static TrainingReportValidationArtifacts.ArtifactVerification verifyTrainingReportValidationArtifacts(
            Path outputDirectory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256) throws IOException {
        return TrainingReportValidationArtifacts.verify(
                outputDirectory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256);
    }

    public static List<TrainingReportDiagnostics.Finding> analyzeTrainingReport(Path reportFile) throws IOException {
        return TrainingReportDiagnostics.analyze(reportFile);
    }

    public static List<TrainingReportDiagnostics.Finding> analyzeTrainingReport(Map<String, ?> report) {
        return TrainingReportDiagnostics.analyze(report);
    }
}
