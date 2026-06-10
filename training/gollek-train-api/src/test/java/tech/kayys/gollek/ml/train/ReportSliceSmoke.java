package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.SmokeAssertions.require;
import static tech.kayys.gollek.ml.train.SmokeAssertions.requireContains;
import static tech.kayys.gollek.ml.train.SmokeAssertions.requireEquals;
import static tech.kayys.gollek.ml.train.SmokeAssertions.requireNear;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public final class ReportSliceSmoke {
    public static void main(String[] args) {
        TrainerRuntimeProfiler profiler = new TrainerRuntimeProfiler();
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 4_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.TRAIN_LOSS, 10_000_000L);
        profiler.record(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP, 7_000_000L);

        Map<String, Object> metadata = profiler.toMetadata("runtimeProfile");
        requireEquals("train", metadata.get("runtimeProfile.primaryGroup.name"), "primary group");
        requireEquals("train.loss", metadata.get("runtimeProfile.primaryHotspot.phase"), "primary hotspot");
        requireNear(14.0, metadata.get("runtimeProfile.primaryGroup.totalMillis"), "primary group total");
        requireNear(66.6666666667, metadata.get("runtimeProfile.primaryGroup.percentTotal"), "primary group percent");
        requireNear(4.0, metadata.get("runtimeProfile.primaryGroup.minMillis"), "primary group min");
        requireNear(10.0, metadata.get("runtimeProfile.primaryGroup.lastMillis"), "primary group last");
        requireNear(3.0, metadata.get("runtimeProfile.primaryGroup.stddevMillis"), "primary group stddev");
        requireNear(47.6190476190, metadata.get("runtimeProfile.primaryHotspot.percentTotal"), "primary hotspot percent");
        requireNear(10.0, metadata.get("runtimeProfile.primaryHotspot.minMillis"), "primary hotspot min");
        requireNear(10.0, metadata.get("runtimeProfile.primaryHotspot.lastMillis"), "primary hotspot last");
        requireNear(0.0, metadata.get("runtimeProfile.primaryHotspot.stddevMillis"), "primary hotspot stddev");

        TrainingReportRuntimeProfile profile = TrainingReportRuntimeProfile.fromMetadata(metadata);
        require(profile.available(), "runtime profile should be available");
        requireEquals("train", profile.primaryGroup().orElseThrow().name(), "typed primary group");
        requireEquals(2, profile.groupCount(), "typed group count");
        requireEquals(3, profile.hotspotCount(), "typed hotspot count");

        String runtimeMarkdown = TrainingReportRuntimeProfileMarkdown.render(profile);
        requireContains(runtimeMarkdown, "**Primary group:** `train` (14.000 ms total)");
        requireContains(runtimeMarkdown, "| Group | Count | Total ms | Total % | Avg ms | Min ms | Max ms | Last ms | Stddev ms |");
        requireContains(runtimeMarkdown, "| `optimizer` | 1 | 7.000 | 33.333 | 7.000 | 7.000 | 7.000 | 7.000 | 0.000 |");
        requireContains(runtimeMarkdown, "| Rank | Phase | Count | Total ms | Total % | Avg ms | Min ms | Max ms | Last ms | Stddev ms |");
        requireContains(runtimeMarkdown, "| 1 | `train.loss` | 1 | 10.000 | 47.619 | 10.000 | 10.000 | 10.000 | 10.000 | 0.000 |");

        TrainingReportParameterUpdateDiagnosticsPolicy policy =
                new TrainingReportParameterUpdateDiagnosticsPolicy(true, true, 8);
        String policyMarkdown = TrainingReportParameterUpdateDiagnosticsPolicyMarkdown.render(policy);
        requireContains(policyMarkdown, "| `yes` | `yes` | 8 |");
        requireContains(policyMarkdown, "sampled every 8 optimizer step(s)");

        Map<String, Object> report = Map.of(
                "diagnostics", List.of(),
                "metadata", metadata);
        TrainingReportActionPlan actionPlan = TrainingReportAdvisor.actionPlan(report);
        require(actionPlan.recommendations().stream()
                .anyMatch(recommendation -> recommendation.diagnosticCode().equals("runtime_profile.primary_hotspot")),
                "missing primary hotspot recommendation");
        TrainingReportRecommendation groupRecommendation = actionPlan.recommendations().stream()
                .filter(recommendation -> recommendation.diagnosticCode().equals("runtime_profile.primary_group"))
                .findFirst()
                .orElseThrow();
        requireEquals("train", groupRecommendation.evidence().get("group"), "advisor group evidence");
        requireNear(66.6666666667, groupRecommendation.evidence().get("percentTotal"), "advisor group percent evidence");
        requireNear(4.0, groupRecommendation.evidence().get("minMillis"), "advisor group min evidence");
        requireNear(10.0, groupRecommendation.evidence().get("lastMillis"), "advisor group last evidence");
        requireNear(3.0, groupRecommendation.evidence().get("stddevMillis"), "advisor group stddev evidence");
        require(groupRecommendation.actions().stream()
                .anyMatch(action -> action.contains("forward, backward, loss, and metric")),
                "missing train group action");
        TrainingReportRecommendation variabilityRecommendation = actionPlan.recommendations().stream()
                .filter(recommendation -> recommendation.diagnosticCode().equals("runtime_profile.primary_group_variability"))
                .findFirst()
                .orElseThrow();
        requireEquals("train", variabilityRecommendation.evidence().get("group"), "variability group evidence");
        requireNear(3.0 / 7.0, variabilityRecommendation.evidence().get("coefficientOfVariation"),
                "variability coefficient evidence");
        requireNear(0.35, variabilityRecommendation.evidence().get("threshold"), "variability threshold evidence");
        require(variabilityRecommendation.actions().stream()
                .anyMatch(action -> action.contains("data-loader and preprocessing jitter")),
                "missing train variability action");
        TrainingReportRecommendation spikeRecommendation = actionPlan.recommendations().stream()
                .filter(recommendation -> recommendation.diagnosticCode().equals("runtime_profile.primary_group_spike"))
                .findFirst()
                .orElseThrow();
        requireEquals("train", spikeRecommendation.evidence().get("group"), "spike group evidence");
        requireNear(10.0 / 7.0, spikeRecommendation.evidence().get("maxToAverageRatio"),
                "spike max-to-average evidence");
        requireNear(1.40, spikeRecommendation.evidence().get("threshold"), "spike threshold evidence");
        require(spikeRecommendation.actions().stream()
                .anyMatch(action -> action.contains("worst train sample")),
                "missing train spike action");

        String actionPlanMarkdown = TrainingReportActionPlanMarkdown.render(actionPlan);
        requireContains(actionPlanMarkdown, "| `MEDIUM` | `TRAINING_DYNAMICS` | `runtime_profile.primary_group`");
        requireContains(actionPlanMarkdown, "| `MEDIUM` | `TRAINING_DYNAMICS` | `runtime_profile.primary_group_variability`");
        requireContains(actionPlanMarkdown, "| `MEDIUM` | `TRAINING_DYNAMICS` | `runtime_profile.primary_group_spike`");
        requireContains(actionPlanMarkdown, "Review trainer runtime group \\`train\\`");
        requireContains(actionPlanMarkdown, "Stabilize trainer runtime group \\`train\\`");
        requireContains(actionPlanMarkdown, "Investigate worst-case trainer runtime spike in \\`train\\`");
        requireContains(actionPlanMarkdown, "Inspect train forward, backward, loss, and metric timings together");

        TrainerRuntimeProfiler baselineProfiler = new TrainerRuntimeProfiler();
        baselineProfiler.record(TrainerRuntimeProfiler.Phase.TRAIN_FORWARD, 2_000_000L);
        baselineProfiler.record(TrainerRuntimeProfiler.Phase.TRAIN_LOSS, 5_000_000L);
        baselineProfiler.record(TrainerRuntimeProfiler.Phase.OPTIMIZER_STEP, 7_000_000L);
        Map<String, Object> baselineReport = Map.of(
                "diagnostics", List.of(),
                "metadata", baselineProfiler.toMetadata("runtimeProfile"));
        TrainingReportRuntimeRegressionSummary regressionSummary =
                TrainingReportAdvisor.runtimeRegressionSummary(baselineReport, report);
        require(regressionSummary.available(), "runtime regression summary should be available");
        require(regressionSummary.regressed(), "runtime regression summary should flag regression");
        requireEquals("train", regressionSummary.primaryGroupAverage().orElseThrow().key(),
                "summary primary group key");
        requireNear(2.0, regressionSummary.primaryGroupAverage().orElseThrow().ratio(),
                "summary primary group ratio");
        requireEquals("train.loss", regressionSummary.primaryHotspotAverage().orElseThrow().key(),
                "summary primary hotspot key");
        requireNear(2.0, regressionSummary.primaryHotspotAverage().orElseThrow().ratio(),
                "summary primary hotspot ratio");
        Map<String, Object> regressionSummaryMap = regressionSummary.toMap();
        requireEquals(true, regressionSummaryMap.get("available"), "summary map available");
        requireEquals(true, regressionSummaryMap.get("regressed"), "summary map regressed");
        requireEquals(regressionSummaryMap, TrainingReportAdvisor.runtimeRegression(baselineReport, report),
                "facade runtime regression map");
        String regressionMarkdown = TrainingReportRuntimeRegressionSummaryMarkdown.render(regressionSummary);
        requireContains(regressionMarkdown, "## Runtime Regression Summary");
        requireContains(regressionMarkdown, "**Regressed:** `yes`");
        requireContains(regressionMarkdown,
                "| Scope | Key | Baseline avg ms | Candidate avg ms | Ratio | Threshold | Regressed |");
        requireContains(regressionMarkdown, "| primary group | `train` | 3.500 | 7.000 | 2.000 | 1.150 | `yes` |");
        requireContains(regressionMarkdown,
                "| primary hotspot | `train.loss` | 5.000 | 10.000 | 2.000 | 1.200 | `yes` |");
        requireEquals(regressionMarkdown, TrainingReportAdvisor.runtimeRegressionMarkdown(baselineReport, report),
                "facade runtime regression markdown");
        TrainingReportActionPlan comparisonPlan = TrainingReportAdvisor.actionPlan(baselineReport, report);
        String comparisonMarkdown = TrainingReportActionPlanMarkdown.render(comparisonPlan, regressionSummary);
        requireContains(comparisonMarkdown, "## Runtime Regression Summary");
        requireContains(comparisonMarkdown,
                "| primary group | `train` | 3.500 | 7.000 | 2.000 | 1.150 | `yes` |");
        requireContains(comparisonMarkdown,
                "| `HIGH` | `TRAINING_DYNAMICS` | `runtime_profile.primary_group_average_regressed`");
        requireEquals(comparisonMarkdown, TrainingReportAdvisor.comparisonActionPlanMarkdown(baselineReport, report),
                "facade comparison action plan markdown");
        TrainingReportComparisonActionPlanExport typedComparisonExport =
                TrainingReportAdvisor.comparisonActionPlanExport(baselineReport, report);
        require(typedComparisonExport.regressed(), "typed comparison export regressed");
        require(typedComparisonExport.requiresAttention(), "typed comparison export requires attention");
        requireEquals(regressionSummaryMap, typedComparisonExport.runtimeRegression().toMap(),
                "typed comparison export runtime regression");
        requireEquals(comparisonMarkdown, typedComparisonExport.markdown(), "typed comparison export markdown");
        Map<String, Object> comparisonExport = TrainingReportAdvisor.comparisonActionPlan(baselineReport, report);
        requireEquals(TrainingReportComparisonActionPlanExport.SCHEMA,
                TrainingReportAdvisor.COMPARISON_ACTION_PLAN_SCHEMA, "comparison export schema constant");
        requireEquals(TrainingReportComparisonActionPlanExport.SCHEMA,
                TrainingReportAdvisor.comparisonActionPlanSchema(), "comparison export schema accessor");
        requireEquals(TrainingReportAdvisor.COMPARISON_ACTION_PLAN_SCHEMA, comparisonExport.get("schema"),
                "comparison export schema");
        require(comparisonExport.get("actionPlan") instanceof Map<?, ?>, "comparison export action plan");
        requireEquals(regressionSummaryMap, comparisonExport.get("runtimeRegression"),
                "comparison export runtime regression");
        requireEquals(comparisonMarkdown, comparisonExport.get("markdown"), "comparison export markdown");
        requireEquals(typedComparisonExport.toMap(), comparisonExport, "typed comparison export map");
        TrainingReportComparisonActionPlanExport roundTrippedComparisonExport =
                TrainingReportComparisonActionPlanExport.fromMap(comparisonExport);
        requireEquals(typedComparisonExport.toMap(), roundTrippedComparisonExport.toMap(),
                "typed comparison export round trip");
        String comparisonExportJson = typedComparisonExport.toJson();
        requireContains(comparisonExportJson, "\"schema\":\"" + TrainingReportComparisonActionPlanExport.SCHEMA + "\"");
        TrainingReportComparisonActionPlanExport jsonRoundTrippedComparisonExport =
                TrainingReportComparisonActionPlanExport.fromJson(comparisonExportJson);
        requireEquals(typedComparisonExport.markdown(), jsonRoundTrippedComparisonExport.markdown(),
                "typed comparison export json markdown round trip");
        requireEquals(typedComparisonExport.runtimeRegression().toMap(),
                jsonRoundTrippedComparisonExport.runtimeRegression().toMap(),
                "typed comparison export json regression round trip");
        requireEquals(typedComparisonExport.actionPlan().status(), jsonRoundTrippedComparisonExport.actionPlan().status(),
                "typed comparison export json status round trip");
        requireEquals(typedComparisonExport.actionPlan().recommendations().size(),
                jsonRoundTrippedComparisonExport.actionPlan().recommendations().size(),
                "typed comparison export json recommendation count round trip");
        try {
            Path exportFile = Files.createTempFile("gollek-comparison-action-plan", ".json");
            Path markdownFile = Files.createTempFile("gollek-comparison-action-plan", ".md");
            typedComparisonExport.writeJson(exportFile);
            typedComparisonExport.writeMarkdown(markdownFile);
            TrainingReportComparisonActionPlanExport fileRoundTrippedComparisonExport =
                    TrainingReportComparisonActionPlanExport.readJson(exportFile);
            requireEquals(typedComparisonExport.markdown(), fileRoundTrippedComparisonExport.markdown(),
                    "typed comparison export file markdown round trip");
            requireEquals(typedComparisonExport.runtimeRegression().toMap(),
                    fileRoundTrippedComparisonExport.runtimeRegression().toMap(),
                    "typed comparison export file regression round trip");
            requireEquals(typedComparisonExport.markdown(),
                    TrainingReportComparisonActionPlanExport.readMarkdown(markdownFile),
                    "typed comparison export markdown file round trip");
            Map<String, Object> artifactManifest = typedComparisonExport.artifactManifest(exportFile, markdownFile);
            requireEquals(TrainingReportComparisonActionPlanExport.ARTIFACT_MANIFEST_SCHEMA,
                    artifactManifest.get("schema"), "comparison export artifact manifest schema");
            requireEquals(TrainingReportComparisonActionPlanExport.SCHEMA,
                    artifactManifest.get("exportSchema"), "comparison export artifact manifest export schema");
            requireEquals(true, artifactManifest.get("regressed"), "comparison export artifact manifest regressed");
            requireEquals(true, artifactManifest.get("requiresAttention"),
                    "comparison export artifact manifest requires attention");
            require(artifactManifest.get("json") instanceof Map<?, ?>, "comparison export artifact manifest json");
            require(artifactManifest.get("markdown") instanceof Map<?, ?>,
                    "comparison export artifact manifest markdown");
            Path bundleDirectory = Files.createTempDirectory("gollek-comparison-action-plan-bundle");
            TrainingReportComparisonActionPlanArtifacts.ArtifactBundle bundle =
                    TrainingReportComparisonActionPlanArtifacts.write(bundleDirectory, typedComparisonExport);
            require(bundle.regressed(), "comparison export artifact bundle regressed");
            require(bundle.requiresAttention(), "comparison export artifact bundle requires attention");
            requireEquals(typedComparisonExport.markdown(),
                    TrainingReportComparisonActionPlanExport.readMarkdown(bundle.markdownFile()),
                    "comparison export artifact bundle markdown");
            requireEquals(typedComparisonExport.runtimeRegression().toMap(),
                    TrainingReportComparisonActionPlanExport.readJson(bundle.jsonFile()).runtimeRegression().toMap(),
                    "comparison export artifact bundle json");
            requireEquals(TrainingReportComparisonActionPlanExport.ARTIFACT_MANIFEST_SCHEMA,
                    bundle.manifest().get("schema"), "comparison export artifact bundle manifest schema");
            require(Files.exists(bundle.manifestFile()), "comparison export artifact bundle manifest file");
            TrainingReportComparisonActionPlanArtifacts.Verification verification =
                    TrainingReportComparisonActionPlanArtifacts.verify(bundle);
            require(verification.passed(), "comparison export artifact bundle verification");
            require(!verification.hasFailures(), "comparison export artifact bundle verification has failures");
            requireEquals(0, verification.failureCount(), "comparison export artifact bundle verification failures");
            requireEquals("Comparison action-plan artifacts verified.", verification.summary(),
                    "comparison export artifact bundle verification summary");
            requireEquals(true, verification.toMap().get("passed"),
                    "comparison export artifact bundle verification map");
            requireEquals(false, verification.toMap().get("hasFailures"),
                    "comparison export artifact bundle verification failure map");
            requireEquals(0, verification.toMap().get("failureCount"),
                    "comparison export artifact bundle verification count map");
            String verificationMarkdown =
                    TrainingReportComparisonActionPlanArtifactVerificationMarkdown.render(verification);
            requireContains(verificationMarkdown, "## Comparison Action-Plan Artifact Verification");
            requireContains(verificationMarkdown, "**Status:** `PASS`");
            requireContains(verificationMarkdown, "**Failures:** `0`");
            Files.writeString(bundle.markdownFile(), "\n# tampered\n", java.nio.file.StandardOpenOption.APPEND);
            TrainingReportComparisonActionPlanArtifacts.Verification tamperedVerification =
                    TrainingReportComparisonActionPlanArtifacts.verify(bundle.manifestFile());
            require(!tamperedVerification.passed(), "comparison export artifact bundle tamper verification");
            require(tamperedVerification.hasFailures(),
                    "comparison export artifact bundle tamper verification has failures");
            require(tamperedVerification.failureCount() > 0,
                    "comparison export artifact bundle tamper verification failure count");
            requireContains(tamperedVerification.summary(), "failed with");
            requireEquals(true, tamperedVerification.toMap().get("hasFailures"),
                    "comparison export artifact bundle tamper failure map");
            String tamperedVerificationMarkdown =
                    TrainingReportComparisonActionPlanArtifactVerificationMarkdown.render(tamperedVerification);
            requireContains(tamperedVerificationMarkdown, "**Status:** `FAIL`");
            requireContains(tamperedVerificationMarkdown, "### Failures");
            requireContains(tamperedVerificationMarkdown, "markdown");
            requireContains(tamperedVerificationMarkdown, "SHA-256");
            require(tamperedVerification.failures().stream()
                    .anyMatch(failure -> failure.contains("markdown") && failure.contains("SHA-256")),
                    "comparison export artifact bundle tamper failure");
            Files.deleteIfExists(exportFile);
            Files.deleteIfExists(markdownFile);
            Files.deleteIfExists(bundle.jsonFile());
            Files.deleteIfExists(bundle.markdownFile());
            Files.deleteIfExists(bundle.manifestFile());
            Files.deleteIfExists(bundle.directory());
        } catch (java.io.IOException error) {
            throw new AssertionError("comparison export file round trip failed", error);
        }
        boolean rejectedBadSchema = false;
        try {
            TrainingReportComparisonActionPlanExport.fromMap(Map.of("schema", "wrong.schema"));
        } catch (IllegalArgumentException expected) {
            rejectedBadSchema = true;
        }
        require(rejectedBadSchema, "comparison export schema should be validated");
        TrainingReportRecommendation groupRegression = comparisonPlan.recommendations().stream()
                .filter(recommendation ->
                        recommendation.diagnosticCode().equals("runtime_profile.primary_group_average_regressed"))
                .findFirst()
                .orElseThrow();
        requireEquals("train", groupRegression.evidence().get("group"), "group regression evidence");
        requireNear(3.5, groupRegression.evidence().get("baselineAverageMillis"),
                "group baseline average evidence");
        requireNear(7.0, groupRegression.evidence().get("candidateAverageMillis"),
                "group candidate average evidence");
        requireNear(2.0, groupRegression.evidence().get("ratio"), "group regression ratio evidence");
        requireNear(1.15, groupRegression.evidence().get("threshold"), "group regression threshold evidence");
        require(groupRegression.actions().stream()
                .anyMatch(action -> action.contains("dynamic shapes")),
                "missing train group regression action");
        TrainingReportRecommendation hotspotRegression = comparisonPlan.recommendations().stream()
                .filter(recommendation ->
                        recommendation.diagnosticCode().equals("runtime_profile.primary_hotspot_average_regressed"))
                .findFirst()
                .orElseThrow();
        requireEquals("train.loss", hotspotRegression.evidence().get("phase"), "hotspot regression evidence");
        requireNear(5.0, hotspotRegression.evidence().get("baselineAverageMillis"),
                "hotspot baseline average evidence");
        requireNear(10.0, hotspotRegression.evidence().get("candidateAverageMillis"),
                "hotspot candidate average evidence");
        requireNear(2.0, hotspotRegression.evidence().get("ratio"), "hotspot regression ratio evidence");
        requireNear(1.20, hotspotRegression.evidence().get("threshold"), "hotspot regression threshold evidence");
    }
}
