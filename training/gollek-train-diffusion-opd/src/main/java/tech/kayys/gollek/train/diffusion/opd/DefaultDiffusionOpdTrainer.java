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

    private final DiffusionOpdConfig config;
    private final DiffusionDenoiser student;
    private final Map<String, DiffusionDenoiser> teachers;
    private final List<DiffusionOpdListener> listeners;
    private final DiffusionConditioningResolver conditioningResolver;
    private final DiffusionOptimizationStep optimizationStep;
    private final DiffusionScheduler scheduler;
    private final TransitionMeanAdapter transitionMeanAdapter;
    private final long[] latentShape;
    private final Path summaryFile;
    private final Path historyFile;
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
        this.latentShape = builder.latentShape.clone();
        this.summaryFile = resolveOutputFile(config.checkpointDir(), SUMMARY_JSON_FILE_NAME);
        this.historyFile = resolveOutputFile(config.checkpointDir(), HISTORY_CSV_FILE_NAME);
        if (config.tasks().isEmpty()) {
            throw new IllegalStateException("Diffusion OPD requires at least one task");
        }
        validateTeachers(config.tasks(), teachers);
        this.latestSummary = new TrainingSummary(
                0,
                Double.NaN,
                -1,
                null,
                null,
                0L,
                Map.of(
                        "runtime", "diffusion-opd-java",
                        "samplerType", config.samplerType().name(),
                        "taskCount", config.tasks().size()));
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
        notifyTrainingStart();

        for (int round = 0; round < config.maxRounds() && !stopped.get(); round++) {
            notifyRoundStart(round);
            double roundLossSum = 0.0;
            int roundLossCount = 0;
            int roundSteps = 0;
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
                            double weightedStepLossValue = baseStepLossValue * resolvedTeacher.lossWeight();
                            optimizationStep.update(stepLoss.mul(resolvedTeacher.lossWeight()));
                            totalLoss += weightedStepLossValue;
                            roundLossSum += weightedStepLossValue;
                            lossCount++;
                            roundLossCount++;
                            totalSteps++;
                            roundSteps++;
                            teacherUsage.merge(resolvedTeacher.teacherKey(), 1, Integer::sum);
                            stageUsage.merge(resolvedTeacher.stageName(), 1, Integer::sum);
                            weightedStageLoss.merge(
                                    resolvedTeacher.stageName(),
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
            recordRoundHistory(round, roundMeanLoss, roundSteps);
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
        metadata.put("teacherBindings", describeTeacherBindings(config.tasks()));
        metadata.put("stopped", stopped.get());
        metadata.put("checkpointDir", String.valueOf(config.checkpointDir()));
        metadata.put("summaryFile", summaryFile == null ? null : summaryFile.toString());
        metadata.put("historyFile", historyFile == null ? null : historyFile.toString());
        metadata.put("roundHistory", List.copyOf(roundHistory));
        latestSummary = new TrainingSummary(
                completedRounds,
                Double.NaN,
                -1,
                latestLoss,
                null,
                durationMs,
                Map.copyOf(metadata));
        persistSummarySafely(latestSummary);
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
        for (DiffusionOpdListener listener : listeners) {
            listener.onStep(this, round, task, timestep, teacherKey, stepLoss);
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

    private void recordRoundHistory(int round, double meanLoss, int optimizationSteps) {
        roundHistory.add(Map.of(
                "round", round,
                "meanLoss", meanLoss,
                "optimizationSteps", optimizationSteps));
        roundHistory.sort(Comparator.comparingInt(entry -> ((Number) entry.get("round")).intValue()));
    }

    private void persistHistorySafely() {
        if (historyFile == null) {
            return;
        }
        try {
            Files.createDirectories(historyFile.getParent());
            List<String> lines = new ArrayList<>();
            lines.add("round,mean_loss,optimization_steps");
            for (Map<String, Object> row : roundHistory) {
                lines.add(row.get("round") + "," + row.get("meanLoss") + "," + row.get("optimizationSteps"));
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

        public Builder listener(DiffusionOpdListener listener) {
            this.listeners.add(Objects.requireNonNull(listener, "listener must not be null"));
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
