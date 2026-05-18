package tech.kayys.gollek.train.diffusion.opd;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.train.diffusion.api.DiffusionConditioningResolver;
import tech.kayys.gollek.train.diffusion.api.DiffusionDenoiser;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdConfig;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdListener;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdArtifactsReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRuntimeObserver;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRunReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.train.diffusion.api.DiffusionOptimizationStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionSamplerType;
import tech.kayys.gollek.train.diffusion.api.DiffusionScheduler;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.train.diffusion.api.DiffusionTeacherBinding;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Java-first scaffold for diffusion on-policy distillation.
 *
 * <p>This implementation intentionally focuses on stable module boundaries:
 * task routing, rollout orchestration, transition-mean supervision, and a
 * top-level builder native to Gollek's training API. It is designed to evolve
 * into the full DiffusionOPD algorithm without forcing Python-side orchestration.
 *
 * <p>Algorithm reference:
 * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
 * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
 */
public final class DefaultDiffusionOpdTrainer implements DiffusionOpdSession {
    private static final String SUMMARY_JSON_FILE_NAME = "diffusion-opd-summary.json";
    private static final String HISTORY_CSV_FILE_NAME = "diffusion-opd-history.csv";
    private static final String REPORT_JSON_FILE_NAME = "diffusion-opd-report.json";

    private final DiffusionOpdConfig config;
    private final DiffusionDenoiser student;
    private final Map<String, DiffusionDenoiser> teachers;
    private final List<DiffusionOpdListener> listeners;
    private final List<DiffusionOpdRuntimeObserver> runtimeObservers;
    private final DiffusionConditioningResolver conditioningResolver;
    private final DiffusionOptimizationStep optimizationStep;
    private final DiffusionScheduler scheduler;
    private final TransitionMeanAdapter transitionMeanAdapter;
    private final boolean adaptiveStageWeighting;
    private final double adaptiveStageWeightMomentum;
    private final double adaptiveStageWeightMinFactor;
    private final double adaptiveStageWeightMaxFactor;
    private final long[] latentShape;
    private final Path summaryFile;
    private final Path historyFile;
    private final Path reportFile;
    private final Map<String, Object> metadata;
    private final Map<String, Object> roundHistoryMetadata;
    private final StageAwareTeacherSelector teacherSelector = new StageAwareTeacherSelector();
    private final AtomicBoolean stopped = new AtomicBoolean(false);
    private final List<Map<String, Object>> roundHistory = new ArrayList<>();

    private volatile TrainingSummary latestSummary;

    private DefaultDiffusionOpdTrainer(Builder builder) {
        this.config = new DiffusionOpdConfig(
                builder.tasks,
                builder.samplerType,
                builder.batchSize,
                builder.gradientAccumulationSteps,
                builder.maxRounds,
                builder.seed,
                builder.checkpointDir);
        this.student = Objects.requireNonNull(builder.student, "student must not be null");
        this.teachers = Map.copyOf(builder.teachers);
        this.listeners = List.copyOf(builder.listeners);
        this.runtimeObservers = List.copyOf(builder.runtimeObservers);
        this.conditioningResolver = Objects.requireNonNull(
                builder.conditioningResolver,
                "conditioningResolver must not be null");
        this.optimizationStep = builder.optimizationStep == null
                ? Tensor::backward
                : builder.optimizationStep;
        this.scheduler = Objects.requireNonNull(builder.scheduler, "scheduler must not be null");
        this.transitionMeanAdapter = builder.transitionMeanAdapter == null
                ? new SchedulerStepTransitionMeanAdapter(scheduler)
                : builder.transitionMeanAdapter;
        this.adaptiveStageWeighting = builder.adaptiveStageWeighting;
        this.adaptiveStageWeightMomentum = builder.adaptiveStageWeightMomentum;
        this.adaptiveStageWeightMinFactor = builder.adaptiveStageWeightMinFactor;
        this.adaptiveStageWeightMaxFactor = builder.adaptiveStageWeightMaxFactor;
        this.latentShape = builder.latentShape.clone();
        this.summaryFile = resolveOutputFile(config.checkpointDir(), SUMMARY_JSON_FILE_NAME);
        this.historyFile = resolveOutputFile(config.checkpointDir(), HISTORY_CSV_FILE_NAME);
        this.reportFile = resolveOutputFile(config.checkpointDir(), REPORT_JSON_FILE_NAME);
        this.metadata = Map.copyOf(builder.metadata);
        this.roundHistoryMetadata = Map.copyOf(builder.roundHistoryMetadata);
        if (config.tasks().isEmpty()) {
            throw new IllegalStateException("Diffusion OPD requires at least one task");
        }
        validateTeachers(config.tasks(), teachers);
        Map<String, Object> initialMetadata = new LinkedHashMap<>();
        initialMetadata.put("runtime", "diffusion-opd-java");
        initialMetadata.put("samplerType", config.samplerType().name());
        initialMetadata.put("taskCount", config.tasks().size());
        initialMetadata.putAll(metadata);
        this.latestSummary = new TrainingSummary(
                0,
                Double.NaN,
                -1,
                null,
                null,
                0L,
                Map.copyOf(initialMetadata));
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public DiffusionOpdConfig config() {
        return config;
    }

    @Override
    public TrainingSummary fit() {
        long startedAt = System.currentTimeMillis();
        Random random = new Random(config.seed());
        RoundRobinTaskSampler sampler = new RoundRobinTaskSampler(config.tasks());
        double totalLoss = 0.0;
        int lossCount = 0;
        int completedRounds = 0;
        int totalSteps = 0;
        Map<String, Integer> teacherUsage = new LinkedHashMap<>();
        Map<String, Integer> stageUsage = new LinkedHashMap<>();
        Map<String, Double> weightedStageLoss = new LinkedHashMap<>();
        Map<String, Integer> taskStageUsage = new LinkedHashMap<>();
        Map<String, Double> taskStageWeightedLoss = new LinkedHashMap<>();
        Map<String, Double> adaptiveTaskStageFactors = initializeAdaptiveTaskStageFactors(config.tasks());
        notifyTrainingStart();

        for (int round = 0; round < config.maxRounds() && !stopped.get(); round++) {
            notifyRoundStart(round);
            double roundLossSum = 0.0;
            int roundLossCount = 0;
            int roundSteps = 0;
            Map<String, Double> roundBaseStageLoss = new LinkedHashMap<>();
            Map<String, Integer> roundStageSteps = new LinkedHashMap<>();
            for (int taskIndex = 0; taskIndex < config.tasks().size() && !stopped.get(); taskIndex++) {
                DiffusionTask task = sampler.next();
                if (task.promptSamples().isEmpty()) {
                    continue;
                }
                notifyTaskStart(round, task);
                int batchLimit = Math.min(config.batchSize(), task.promptSamples().size());
                for (int i = 0; i < batchLimit && !stopped.get(); i++) {
                    DiffusionPromptSample sample = pickSample(task.promptSamples(), random, i);
                    Tensor conditioning = conditioningResolver.resolve(sample);
                    notifyConditioningResolved(round, task, sample, conditioning);
                    Tensor latents = Tensor.randn(latentShape);
                    try {
                        for (int tIndex = 0; tIndex < scheduler.timesteps().length && !stopped.get(); tIndex++) {
                            int timestep = scheduler.timesteps()[tIndex];
                            StageAwareTeacherSelector.ResolvedTeacher resolvedTeacher =
                                    teacherSelector.resolve(task, tIndex, teachers);
                            Tensor studentPrediction = student.predict(latents, conditioning, timestep);
                            Tensor teacherPrediction = resolvedTeacher.teacher().predict(latents, conditioning, timestep);
                            Tensor studentMean = transitionMeanAdapter.transitionMean(latents, studentPrediction, tIndex);
                            Tensor teacherMean = transitionMeanAdapter.transitionMean(latents, teacherPrediction, tIndex);
                            Tensor stepLoss = DiffusionOpdLosses.meanMatchingLoss(
                                    studentMean,
                                    teacherMean,
                                    config.samplerType(),
                                    transitionMeanAdapter.stepVariance(tIndex));
                            double baseStepLossValue = stepLoss.item();
                            String taskStageKey = taskStageKey(task.id(), resolvedTeacher.stageName());
                            double effectiveStageWeight = effectiveStageWeight(
                                    taskStageKey,
                                    resolvedTeacher,
                                    adaptiveTaskStageFactors);
                            double weightedStepLossValue = baseStepLossValue * effectiveStageWeight;
                            optimizationStep.update(stepLoss.mul((float) effectiveStageWeight));
                            totalLoss += weightedStepLossValue;
                            roundLossSum += weightedStepLossValue;
                            lossCount++;
                            roundLossCount++;
                            totalSteps++;
                            roundSteps++;
                            teacherUsage.merge(resolvedTeacher.teacherKey(), 1, Integer::sum);
                            stageUsage.merge(resolvedTeacher.stageName(), 1, Integer::sum);
                            taskStageUsage.merge(taskStageKey, 1, Integer::sum);
                            roundBaseStageLoss.merge(
                                    taskStageKey,
                                    baseStepLossValue,
                                    Double::sum);
                            roundStageSteps.merge(taskStageKey, 1, Integer::sum);
                            weightedStageLoss.merge(
                                    resolvedTeacher.stageName(),
                                    weightedStepLossValue,
                                    Double::sum);
                            taskStageWeightedLoss.merge(
                                    taskStageKey,
                                    weightedStepLossValue,
                                    Double::sum);
                            notifyStep(round, task, timestep, resolvedTeacher.teacherKey(), weightedStepLossValue);
                            latents = studentMean;
                        }
                    } finally {
                        conditioning.release();
                        latents.release();
                    }
                }
            }
            double roundMeanLoss = roundLossCount == 0 ? Double.NaN : roundLossSum / roundLossCount;
            adaptStageWeights(roundBaseStageLoss, roundStageSteps, adaptiveTaskStageFactors);
            recordRoundHistory(
                    round,
                    roundMeanLoss,
                    roundSteps,
                    adaptiveTaskStageFactors,
                    roundBaseStageLoss,
                    roundStageSteps);
            persistHistorySafely();
            notifyRoundEnd(round, roundMeanLoss, roundSteps);
            completedRounds++;
        }

        long durationMs = Math.max(0L, System.currentTimeMillis() - startedAt);
        Double latestLoss = lossCount == 0 ? null : totalLoss / lossCount;
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("runtime", "diffusion-opd-java");
        metadata.put("samplerType", config.samplerType().name());
        metadata.put("taskCount", config.tasks().size());
        metadata.put("roundsCompleted", completedRounds);
        metadata.put("optimizationSteps", totalSteps);
        metadata.put("teacherUsage", Map.copyOf(teacherUsage));
        metadata.put("stageUsage", Map.copyOf(stageUsage));
        metadata.put("stageWeightedLoss", Map.copyOf(weightedStageLoss));
        metadata.put("taskStageUsage", Map.copyOf(taskStageUsage));
        metadata.put("taskStageWeightedLoss", Map.copyOf(taskStageWeightedLoss));
        metadata.put("adaptiveStageWeighting", adaptiveStageWeighting);
        metadata.put("adaptiveStageFactors", aggregateStageFactors(adaptiveTaskStageFactors));
        metadata.put("adaptiveTaskStageFactors", Map.copyOf(adaptiveTaskStageFactors));
        metadata.put("teacherBindings", describeTeacherBindings(config.tasks()));
        metadata.put("stopped", stopped.get());
        metadata.put("checkpointDir", String.valueOf(config.checkpointDir()));
        metadata.put("summaryFile", summaryFile == null ? null : summaryFile.toString());
        metadata.put("historyFile", historyFile == null ? null : historyFile.toString());
        metadata.put("reportFile", reportFile == null ? null : reportFile.toString());
        metadata.put("roundHistory", List.copyOf(roundHistory));
        metadata.putAll(this.metadata);
        metadata.putAll(collectRuntimeSummaryMetadata());
        latestSummary = new TrainingSummary(
                completedRounds,
                Double.NaN,
                -1,
                latestLoss,
                null,
                durationMs,
                Map.copyOf(metadata));
        persistSummarySafely(latestSummary);
        persistReportSafely(latestSummary);
        notifyTrainingEnd(latestSummary);
        return latestSummary;
    }

    @Override
    public TrainingSummary summary() {
        return latestSummary;
    }

    @Override
    public boolean isStopped() {
        return stopped.get();
    }

    @Override
    public void stop() {
        stopped.set(true);
    }

    private static Path resolveOutputFile(Path checkpointDir, String fileName) {
        return checkpointDir == null ? null : checkpointDir.resolve(fileName);
    }

    private void notifyTrainingStart() {
        for (DiffusionOpdListener listener : listeners) {
            listener.onTrainingStart(this);
        }
    }

    private void notifyRoundStart(int round) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onRoundStart(this, round);
        }
    }

    private void notifyTaskStart(int round, DiffusionTask task) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onTaskStart(this, round, task);
        }
    }

    private void notifyStep(int round, DiffusionTask task, int timestep, String teacherKey, double stepLoss) {
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            observer.onStep(this, round, task, timestep, teacherKey, stepLoss);
        }
        for (DiffusionOpdListener listener : listeners) {
            listener.onStep(this, round, task, timestep, teacherKey, stepLoss);
        }
    }

    private void notifyConditioningResolved(
            int round,
            DiffusionTask task,
            DiffusionPromptSample sample,
            Tensor conditioning) {
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            observer.onConditioningResolved(this, round, task, sample, conditioning);
        }
    }

    private void notifyRoundEnd(int round, double meanLoss, int optimizationSteps) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onRoundEnd(this, round, meanLoss, optimizationSteps);
        }
    }

    private void notifyTrainingEnd(TrainingSummary summary) {
        for (DiffusionOpdListener listener : listeners) {
            listener.onTrainingEnd(this, summary);
        }
    }

    private void recordRoundHistory(
            int round,
            double meanLoss,
            int optimizationSteps,
            Map<String, Double> adaptiveTaskStageFactors,
            Map<String, Double> roundBaseStageLoss,
            Map<String, Integer> roundStageSteps) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("round", round);
        row.put("meanLoss", meanLoss);
        row.put("optimizationSteps", optimizationSteps);
        row.put("adaptiveTaskStageFactors", Map.copyOf(adaptiveTaskStageFactors));
        row.put("adaptiveStageFactors", aggregateStageFactors(adaptiveTaskStageFactors));
        row.put("roundBaseStageMeanLoss", meanStageLoss(roundBaseStageLoss, roundStageSteps));
        row.putAll(roundHistoryMetadata);
        row.putAll(collectRuntimeRoundHistoryMetadata());
        roundHistory.add(Map.copyOf(row));
        roundHistory.sort(Comparator.comparingInt(entry -> ((Number) entry.get("round")).intValue()));
    }

    private void persistHistorySafely() {
        if (historyFile == null) {
            return;
        }
        try {
            Files.createDirectories(historyFile.getParent());
            List<String> lines = new ArrayList<>();
            List<String> extraKeys = historyExtraKeys();
            List<String> headers = new ArrayList<>(List.of(
                    "round",
                    "mean_loss",
                    "optimization_steps",
                    "adaptive_stage_factors",
                    "round_base_stage_mean_loss"));
            headers.addAll(extraKeys);
            lines.add(String.join(",", headers));
            for (Map<String, Object> row : roundHistory) {
                List<String> values = new ArrayList<>();
                values.add(String.valueOf(row.get("round")));
                values.add(String.valueOf(row.get("meanLoss")));
                values.add(String.valueOf(row.get("optimizationSteps")));
                values.add(csvEscape(String.valueOf(row.get("adaptiveTaskStageFactors"))));
                values.add(csvEscape(String.valueOf(row.get("roundBaseStageMeanLoss"))));
                for (String key : extraKeys) {
                    values.add(csvEscape(String.valueOf(row.get(key))));
                }
                lines.add(String.join(",", values));
            }
            Files.write(
                    historyFile,
                    lines,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
        } catch (IOException ignored) {
            // Keep the training loop resilient while the persistence contract is
            // still evolving.
        }
    }

    private void persistSummarySafely(TrainingSummary summary) {
        if (summaryFile == null) {
            return;
        }
        try {
            Files.createDirectories(summaryFile.getParent());
            Files.writeString(
                    summaryFile,
                    toJson(summary),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
        } catch (IOException ignored) {
            // Keep the training loop resilient while the persistence contract is
            // still evolving.
        }
    }

    private void persistReportSafely(TrainingSummary summary) {
        if (reportFile == null) {
            return;
        }
        try {
            Files.createDirectories(reportFile.getParent());
            Files.writeString(
                    reportFile,
                    toReportJson(summary),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
        } catch (IOException ignored) {
            // Keep the training loop resilient while the persistence contract is
            // still evolving.
        }
    }

    private static Map<String, List<Map<String, Object>>> describeTeacherBindings(List<DiffusionTask> tasks) {
        Map<String, List<Map<String, Object>>> bindingsByTask = new LinkedHashMap<>();
        for (DiffusionTask task : tasks) {
            List<Map<String, Object>> bindings = new ArrayList<>();
            for (DiffusionTeacherBinding binding : task.teacherBindings()) {
                bindings.add(Map.of(
                        "teacherKey", binding.teacherKey(),
                        "stageName", binding.stageName(),
                        "startStepInclusive", binding.startStepInclusive(),
                        "endStepExclusive", binding.endStepExclusive(),
                        "lossWeight", binding.lossWeight()));
            }
            bindingsByTask.put(task.id(), List.copyOf(bindings));
        }
        return Map.copyOf(bindingsByTask);
    }

    private Map<String, Object> collectRuntimeSummaryMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            metadata.putAll(observer.summaryMetadata());
        }
        return Map.copyOf(metadata);
    }

    private Map<String, Object> collectRuntimeRoundHistoryMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        for (DiffusionOpdRuntimeObserver observer : runtimeObservers) {
            metadata.putAll(observer.roundHistoryMetadata());
        }
        return Map.copyOf(metadata);
    }

    private List<String> historyExtraKeys() {
        List<String> keys = new ArrayList<>();
        for (Map<String, Object> row : roundHistory) {
            for (String key : row.keySet()) {
                if (isDefaultHistoryKey(key) || keys.contains(key)) {
                    continue;
                }
                keys.add(key);
            }
        }
        return List.copyOf(keys);
    }

    private static boolean isDefaultHistoryKey(String key) {
        return switch (key) {
            case "round", "meanLoss", "optimizationSteps", "adaptiveTaskStageFactors", "adaptiveStageFactors", "roundBaseStageMeanLoss" -> true;
            default -> false;
        };
    }

    private Map<String, Double> initializeAdaptiveTaskStageFactors(List<DiffusionTask> tasks) {
        Map<String, Double> factors = new LinkedHashMap<>();
        for (DiffusionTask task : tasks) {
            if (task.teacherBindings().isEmpty()) {
                factors.putIfAbsent(taskStageKey(task.id(), "default"), 1.0d);
                continue;
            }
            for (DiffusionTeacherBinding binding : task.teacherBindings()) {
                factors.putIfAbsent(taskStageKey(task.id(), binding.stageName()), 1.0d);
            }
        }
        return factors;
    }

    private double effectiveStageWeight(
            String taskStageKey,
            StageAwareTeacherSelector.ResolvedTeacher resolvedTeacher,
            Map<String, Double> adaptiveTaskStageFactors) {
        double adaptiveFactor = adaptiveStageWeighting
                ? adaptiveTaskStageFactors.getOrDefault(taskStageKey, 1.0d)
                : 1.0d;
        return resolvedTeacher.lossWeight() * adaptiveFactor;
    }

    private void adaptStageWeights(
            Map<String, Double> roundBaseStageLoss,
            Map<String, Integer> roundStageSteps,
            Map<String, Double> adaptiveTaskStageFactors) {
        if (!adaptiveStageWeighting || roundBaseStageLoss.isEmpty()) {
            return;
        }
        Map<String, Double> stageMeanLoss = meanStageLoss(roundBaseStageLoss, roundStageSteps);
        double globalMean = 0.0d;
        for (double meanLoss : stageMeanLoss.values()) {
            globalMean += meanLoss;
        }
        globalMean /= stageMeanLoss.size();
        if (!Double.isFinite(globalMean) || globalMean <= 0.0d) {
            return;
        }
        for (Map.Entry<String, Double> entry : stageMeanLoss.entrySet()) {
            double currentFactor = adaptiveTaskStageFactors.getOrDefault(entry.getKey(), 1.0d);
            double targetFactor = clamp(entry.getValue() / globalMean, adaptiveStageWeightMinFactor, adaptiveStageWeightMaxFactor);
            double blendedFactor = (adaptiveStageWeightMomentum * currentFactor)
                    + ((1.0d - adaptiveStageWeightMomentum) * targetFactor);
            adaptiveTaskStageFactors.put(
                    entry.getKey(),
                    clamp(blendedFactor, adaptiveStageWeightMinFactor, adaptiveStageWeightMaxFactor));
        }
    }

    private static Map<String, Double> aggregateStageFactors(Map<String, Double> adaptiveTaskStageFactors) {
        Map<String, Double> sums = new LinkedHashMap<>();
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : adaptiveTaskStageFactors.entrySet()) {
            String stageName = stageNameFromTaskStageKey(entry.getKey());
            sums.merge(stageName, entry.getValue(), Double::sum);
            counts.merge(stageName, 1, Integer::sum);
        }
        Map<String, Double> aggregate = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : sums.entrySet()) {
            int count = Math.max(1, counts.getOrDefault(entry.getKey(), 1));
            aggregate.put(entry.getKey(), entry.getValue() / count);
        }
        return Map.copyOf(aggregate);
    }

    private static Map<String, Double> meanStageLoss(
            Map<String, Double> stageLoss,
            Map<String, Integer> stageSteps) {
        Map<String, Double> meanLoss = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : stageLoss.entrySet()) {
            int steps = Math.max(1, stageSteps.getOrDefault(entry.getKey(), 1));
            meanLoss.put(entry.getKey(), entry.getValue() / steps);
        }
        return Map.copyOf(meanLoss);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static String taskStageKey(String taskId, String stageName) {
        return taskId + "::" + stageName;
    }

    private static String stageNameFromTaskStageKey(String taskStageKey) {
        int separator = taskStageKey.indexOf("::");
        return separator >= 0 ? taskStageKey.substring(separator + 2) : taskStageKey;
    }

    private static String csvEscape(String value) {
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }

    private String toJson(TrainingSummary summary) {
        return "{\n"
                + "  \"epochCount\": " + summary.epochCount() + ",\n"
                + "  \"bestValidationLoss\": " + summary.bestValidationLoss() + ",\n"
                + "  \"bestValidationEpoch\": " + summary.bestValidationEpoch() + ",\n"
                + "  \"latestTrainLoss\": " + summary.latestTrainLoss() + ",\n"
                + "  \"latestValidationLoss\": " + summary.latestValidationLoss() + ",\n"
                + "  \"durationMs\": " + summary.durationMs() + ",\n"
                + "  \"metadata\": " + metadataToJson(summary.metadata()) + "\n"
                + "}\n";
    }

    private String toReportJson(TrainingSummary summary) {
        Map<String, Object> metadata = summary.metadata();
        DiffusionOpdRunReport run = new DiffusionOpdRunReport(
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.durationMs(),
                metadata.get("samplerType"),
                metadata.get("taskCount"),
                metadata.get("optimizationSteps"),
                metadata.get("roundsCompleted"),
                metadata.get("stopped"));
        DiffusionOpdArtifactsReport artifacts = new DiffusionOpdArtifactsReport(
                String.valueOf(metadata.get("summaryFile")),
                String.valueOf(metadata.get("historyFile")),
                String.valueOf(metadata.get("reportFile")),
                String.valueOf(metadata.get("checkpointDir")));
        DiffusionOpdReport report = new DiffusionOpdReport(
                run,
                artifacts,
                extractMatching(metadata, "teacher"),
                extractMatching(metadata, "stage"),
                extractMatching(metadata, "task"),
                extractMatching(metadata, "conditioning"),
                extractMatching(metadata, "adaptive"),
                Map.of("teacherBindings", metadata.get("teacherBindings")),
                List.copyOf(roundHistory));
        return valueToJson(report.asMap());
    }

    private static Map<String, Object> extractMatching(Map<String, Object> metadata, String needle) {
        Map<String, Object> extracted = new LinkedHashMap<>();
        String lowerNeedle = needle.toLowerCase();
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            if (entry.getKey().toLowerCase().contains(lowerNeedle)) {
                extracted.put(entry.getKey(), entry.getValue());
            }
        }
        return Map.copyOf(extracted);
    }

    private String metadataToJson(Map<String, Object> metadata) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            if (!first) {
                sb.append(", ");
            }
            first = false;
            sb.append("\"").append(entry.getKey()).append("\": ");
            sb.append(valueToJson(entry.getValue()));
        }
        sb.append("}");
        return sb.toString();
    }

    private String valueToJson(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof Number || value instanceof Boolean) {
            return String.valueOf(value);
        }
        if (value instanceof Map<?, ?> map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> typed = (Map<String, Object>) map;
            return metadataToJson(typed);
        }
        if (value instanceof List<?> list) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(valueToJson(list.get(i)));
            }
            sb.append("]");
            return sb.toString();
        }
        return "\"" + String.valueOf(value).replace("\"", "\\\"") + "\"";
    }

    private static void validateTeachers(List<DiffusionTask> tasks, Map<String, DiffusionDenoiser> teachers) {
        for (DiffusionTask task : tasks) {
            if (task.teacherBindings().isEmpty()) {
                if (!teachers.containsKey(task.id())) {
                    throw new IllegalStateException("Missing teacher for diffusion task: " + task.id());
                }
                continue;
            }
            for (var binding : task.teacherBindings()) {
                if (!teachers.containsKey(binding.teacherKey())) {
                    throw new IllegalStateException(
                            "Missing stage-aware teacher " + binding.teacherKey() + " for task " + task.id());
                }
            }
        }
    }

    private static DiffusionPromptSample pickSample(List<DiffusionPromptSample> samples, Random random, int fallbackIndex) {
        if (samples.size() == 1) {
            return samples.getFirst();
        }
        int index = random.nextInt(samples.size());
        if (index < 0 || index >= samples.size()) {
            index = Math.min(fallbackIndex, samples.size() - 1);
        }
        return samples.get(index);
    }

    public static final class Builder {
        private final List<DiffusionTask> tasks = new ArrayList<>();
        private final Map<String, DiffusionDenoiser> teachers = new LinkedHashMap<>();
        private final List<DiffusionOpdListener> listeners = new ArrayList<>();
        private final List<DiffusionOpdRuntimeObserver> runtimeObservers = new ArrayList<>();
        private DiffusionSamplerType samplerType = DiffusionSamplerType.ODE;
        private DiffusionDenoiser student;
        private DiffusionConditioningResolver conditioningResolver;
        private DiffusionOptimizationStep optimizationStep;
        private DiffusionScheduler scheduler;
        private TransitionMeanAdapter transitionMeanAdapter;
        private int batchSize = 1;
        private int gradientAccumulationSteps = 1;
        private int maxRounds = 1;
        private long seed = 42L;
        private Path checkpointDir;
        private long[] latentShape = new long[] {1, 4, 64, 64};
        private boolean adaptiveStageWeighting;
        private double adaptiveStageWeightMomentum = 0.5d;
        private double adaptiveStageWeightMinFactor = 0.75d;
        private double adaptiveStageWeightMaxFactor = 1.50d;
        private final Map<String, Object> metadata = new LinkedHashMap<>();
        private final Map<String, Object> roundHistoryMetadata = new LinkedHashMap<>();

        private Builder() {
        }

        public Builder samplerType(DiffusionSamplerType samplerType) {
            this.samplerType = Objects.requireNonNull(samplerType, "samplerType must not be null");
            return this;
        }

        public Builder student(DiffusionDenoiser student) {
            this.student = Objects.requireNonNull(student, "student must not be null");
            return this;
        }

        public Builder teacher(String taskId, DiffusionDenoiser teacher) {
            this.teachers.put(
                    Objects.requireNonNull(taskId, "taskId must not be null"),
                    Objects.requireNonNull(teacher, "teacher must not be null"));
            return this;
        }

        public Builder task(DiffusionTask task) {
            this.tasks.add(Objects.requireNonNull(task, "task must not be null"));
            return this;
        }

        public Builder tasks(List<DiffusionTask> tasks) {
            this.tasks.clear();
            this.tasks.addAll(Objects.requireNonNull(tasks, "tasks must not be null"));
            return this;
        }

        public Builder conditioningResolver(DiffusionConditioningResolver conditioningResolver) {
            this.conditioningResolver = Objects.requireNonNull(
                    conditioningResolver,
                    "conditioningResolver must not be null");
            return this;
        }

        public Builder optimizationStep(DiffusionOptimizationStep optimizationStep) {
            this.optimizationStep = Objects.requireNonNull(
                    optimizationStep,
                    "optimizationStep must not be null");
            return this;
        }

        public Builder scheduler(DiffusionScheduler scheduler) {
            this.scheduler = Objects.requireNonNull(scheduler, "scheduler must not be null");
            return this;
        }

        public Builder transitionMeanAdapter(TransitionMeanAdapter transitionMeanAdapter) {
            this.transitionMeanAdapter = Objects.requireNonNull(
                    transitionMeanAdapter,
                    "transitionMeanAdapter must not be null");
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = Math.max(1, batchSize);
            return this;
        }

        public Builder gradientAccumulationSteps(int gradientAccumulationSteps) {
            this.gradientAccumulationSteps = Math.max(1, gradientAccumulationSteps);
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = Math.max(1, maxRounds);
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder checkpointDir(Path checkpointDir) {
            this.checkpointDir = checkpointDir;
            return this;
        }

        public Builder adaptiveStageWeighting(boolean adaptiveStageWeighting) {
            this.adaptiveStageWeighting = adaptiveStageWeighting;
            return this;
        }

        public Builder adaptiveStageWeightMomentum(double adaptiveStageWeightMomentum) {
            this.adaptiveStageWeightMomentum = clamp(adaptiveStageWeightMomentum, 0.0d, 0.999d);
            return this;
        }

        public Builder adaptiveStageWeightRange(double minFactor, double maxFactor) {
            if (!Double.isFinite(minFactor) || !Double.isFinite(maxFactor) || minFactor <= 0.0d || maxFactor < minFactor) {
                throw new IllegalArgumentException("adaptive stage weight range must be finite, > 0, and max >= min");
            }
            this.adaptiveStageWeightMinFactor = minFactor;
            this.adaptiveStageWeightMaxFactor = maxFactor;
            return this;
        }

        public Builder listener(DiffusionOpdListener listener) {
            this.listeners.add(Objects.requireNonNull(listener, "listener must not be null"));
            return this;
        }

        public Builder runtimeObserver(DiffusionOpdRuntimeObserver observer) {
            this.runtimeObservers.add(Objects.requireNonNull(observer, "runtime observer must not be null"));
            return this;
        }

        public Builder runtimeObservers(Iterable<? extends DiffusionOpdRuntimeObserver> observers) {
            Objects.requireNonNull(observers, "runtime observers must not be null");
            for (DiffusionOpdRuntimeObserver observer : observers) {
                runtimeObserver(observer);
            }
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(
                    Objects.requireNonNull(key, "metadata key must not be null"),
                    value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(Objects.requireNonNull(metadata, "metadata must not be null"));
            return this;
        }

        public Builder roundHistoryMetadata(String key, Object value) {
            this.roundHistoryMetadata.put(
                    Objects.requireNonNull(key, "round history metadata key must not be null"),
                    value);
            return this;
        }

        public Builder roundHistoryMetadata(Map<String, Object> metadata) {
            this.roundHistoryMetadata.putAll(Objects.requireNonNull(metadata, "round history metadata must not be null"));
            return this;
        }

        public Builder latentShape(long... latentShape) {
            this.latentShape = Objects.requireNonNull(latentShape, "latentShape must not be null").clone();
            return this;
        }

        public DefaultDiffusionOpdTrainer build() {
            return new DefaultDiffusionOpdTrainer(this);
        }
    }
}
