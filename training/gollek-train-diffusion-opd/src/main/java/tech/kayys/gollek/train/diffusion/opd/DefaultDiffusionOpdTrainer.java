package tech.kayys.gollek.train.diffusion.opd;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdRuntimeObserver;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdSession;
import tech.kayys.gollek.train.diffusion.api.DiffusionOptimizationStep;
import tech.kayys.gollek.train.diffusion.api.DiffusionPromptSample;
import tech.kayys.gollek.train.diffusion.api.DiffusionSamplerType;
import tech.kayys.gollek.train.diffusion.api.DiffusionScheduler;
import tech.kayys.gollek.train.diffusion.api.DiffusionTask;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Java-first scaffold for diffusion on-policy distillation.
 *
 * <p>This implementation intentionally focuses on stable module boundaries:
 * task routing, rollout orchestration, transition-mean supervision, and a
 * top-level builder native to Gollek's training API. It is designed to evolve
 * into the full DiffusionOPD algorithm without forcing Python-side orchestration.
 *
 * <p>The trainer now keeps the orchestration loop in one place and delegates
 * persistence, runtime coordination, and adaptive weighting details to focused
 * package-private helpers so the main session class stays readable.
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
        this.summaryFile = config.checkpointDir() == null ? null : config.checkpointDir().resolve(SUMMARY_JSON_FILE_NAME);
        this.historyFile = config.checkpointDir() == null ? null : config.checkpointDir().resolve(HISTORY_CSV_FILE_NAME);
        this.reportFile = config.checkpointDir() == null ? null : config.checkpointDir().resolve(REPORT_JSON_FILE_NAME);
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

    /**
     * Creates a builder for configuring one Java-native DiffusionOPD training session.
     */
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
        Map<String, Double> adaptiveTaskStageFactors =
                DefaultDiffusionOpdTrainerAdaptiveSupport.initializeAdaptiveTaskStageFactors(config.tasks());
        DefaultDiffusionOpdTrainerRuntimeSupport.notifyTrainingStart(this, listeners);

        for (int round = 0; round < config.maxRounds() && !stopped.get(); round++) {
            DefaultDiffusionOpdTrainerRuntimeSupport.notifyRoundStart(this, listeners, round);
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
                DefaultDiffusionOpdTrainerRuntimeSupport.notifyTaskStart(this, listeners, round, task);
                int batchLimit = Math.min(config.batchSize(), task.promptSamples().size());
                for (int i = 0; i < batchLimit && !stopped.get(); i++) {
                    DiffusionPromptSample sample = pickSample(task.promptSamples(), random, i);
                    Tensor conditioning = conditioningResolver.resolve(sample);
                    DefaultDiffusionOpdTrainerRuntimeSupport.notifyConditioningResolved(
                            this,
                            runtimeObservers,
                            round,
                            task,
                            sample,
                            conditioning);
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
                            String taskStageKey =
                                    DefaultDiffusionOpdTrainerAdaptiveSupport.taskStageKey(task.id(), resolvedTeacher.stageName());
                            double effectiveStageWeight = DefaultDiffusionOpdTrainerAdaptiveSupport.effectiveStageWeight(
                                    adaptiveStageWeighting,
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
                            DefaultDiffusionOpdTrainerRuntimeSupport.notifyStep(
                                    this,
                                    runtimeObservers,
                                    listeners,
                                    round,
                                    task,
                                    timestep,
                                    resolvedTeacher.teacherKey(),
                                    weightedStepLossValue);
                            latents = studentMean;
                        }
                    } finally {
                        conditioning.release();
                        latents.release();
                    }
                }
            }
            double roundMeanLoss = roundLossCount == 0 ? Double.NaN : roundLossSum / roundLossCount;
            DefaultDiffusionOpdTrainerAdaptiveSupport.adaptStageWeights(
                    adaptiveStageWeighting,
                    adaptiveStageWeightMomentum,
                    adaptiveStageWeightMinFactor,
                    adaptiveStageWeightMaxFactor,
                    roundBaseStageLoss,
                    roundStageSteps,
                    adaptiveTaskStageFactors);
            DefaultDiffusionOpdTrainerAdaptiveSupport.recordRoundHistory(
                    roundHistory,
                    roundHistoryMetadata,
                    DefaultDiffusionOpdTrainerRuntimeSupport.collectRuntimeRoundHistoryMetadata(runtimeObservers),
                    round,
                    roundMeanLoss,
                    roundSteps,
                    adaptiveTaskStageFactors,
                    roundBaseStageLoss,
                    roundStageSteps);
            DefaultDiffusionOpdTrainerPersistence.persistHistory(historyFile, roundHistory);
            DefaultDiffusionOpdTrainerRuntimeSupport.notifyRoundEnd(
                    this,
                    listeners,
                    round,
                    roundMeanLoss,
                    roundSteps);
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
        metadata.put("adaptiveStageFactors",
                DefaultDiffusionOpdTrainerAdaptiveSupport.aggregateStageFactors(adaptiveTaskStageFactors));
        metadata.put("adaptiveTaskStageFactors", Map.copyOf(adaptiveTaskStageFactors));
        metadata.put("teacherBindings", DefaultDiffusionOpdTrainerRuntimeSupport.describeTeacherBindings(config.tasks()));
        metadata.put("stopped", stopped.get());
        metadata.put("checkpointDir", String.valueOf(config.checkpointDir()));
        metadata.put("summaryFile", summaryFile == null ? null : summaryFile.toString());
        metadata.put("historyFile", historyFile == null ? null : historyFile.toString());
        metadata.put("reportFile", reportFile == null ? null : reportFile.toString());
        metadata.put("roundHistory", List.copyOf(roundHistory));
        metadata.putAll(this.metadata);
        metadata.putAll(DefaultDiffusionOpdTrainerRuntimeSupport.collectRuntimeSummaryMetadata(runtimeObservers));
        latestSummary = new TrainingSummary(
                completedRounds,
                Double.NaN,
                -1,
                latestLoss,
                null,
                durationMs,
                Map.copyOf(metadata));
        DefaultDiffusionOpdTrainerPersistence.persistSummary(summaryFile, latestSummary);
        DefaultDiffusionOpdTrainerPersistence.persistReport(reportFile, latestSummary, roundHistory);
        DefaultDiffusionOpdTrainerRuntimeSupport.notifyTrainingEnd(this, listeners, latestSummary);
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

    /**
     * Fluent configuration surface for one DiffusionOPD trainer session, covering model wiring,
     * task routing, runtime observers, checkpoint outputs, and adaptive weighting policy.
     */
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
            this.adaptiveStageWeightMomentum =
                    DefaultDiffusionOpdTrainerAdaptiveSupport.clamp(adaptiveStageWeightMomentum, 0.0d, 0.999d);
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

        /**
         * Materializes the configured trainer session using the current builder state and defaulted
         * helper boundaries.
         */
        public DefaultDiffusionOpdTrainer build() {
            return new DefaultDiffusionOpdTrainer(this);
        }
    }
}
