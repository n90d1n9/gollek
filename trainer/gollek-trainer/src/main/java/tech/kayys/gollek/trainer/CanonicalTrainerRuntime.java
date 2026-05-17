package tech.kayys.gollek.trainer;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import tech.kayys.gollek.trainer.api.TrainerConfig;
import tech.kayys.gollek.trainer.api.TrainerRuntimeMode;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;
import tech.kayys.gollek.trainer.api.TrainingSummary;

/**
 * Minimal canonical trainer runtime used when the legacy runtime bridge is not
 * available on the classpath.
 */
public final class CanonicalTrainerRuntime implements TrainerSession {

    private static final String CHECKPOINT_FILE_NAME = "canonical-runtime.state";
    private static final int CHECKPOINT_FORMAT_VERSION = 1;
    private static final String CHECKPOINT_FORMAT_VERSION_KEY = "formatVersion";
    private static final String LEGACY_CHECKPOINT_FORMAT = "legacy-properties";

    private final TrainerConfig config;
    private final List<TrainingListener> listeners;
    private final BatchLossEvaluator trainBatchLossEvaluator;
    private final BatchLossEvaluator validationBatchLossEvaluator;
    private final EpochLossEvaluator trainEpochLossEvaluator;
    private final EpochLossEvaluator validationEpochLossEvaluator;
    private final int earlyStoppingPatience;
    private final double earlyStoppingMinDelta;
    private final boolean resumeFromCheckpoint;
    private final boolean failOnCheckpointLoadError;
    private final Path checkpointStateFile;
    private final CheckpointLoadReport checkpointLoadReport;
    private final CheckpointState initialCheckpointState;
    private final AtomicBoolean stopped = new AtomicBoolean(false);

    private int currentEpoch;
    private int globalStep;
    private int listenerErrorCount;
    private TrainingSummary summary;

    private CanonicalTrainerRuntime(Builder builder) {
        this.config = new TrainerConfig(
                builder.epochs,
                builder.gradientClip,
                builder.mixedPrecision,
                builder.checkpointDir);
        this.listeners = List.copyOf(builder.listeners);
        this.trainBatchLossEvaluator = builder.trainBatchLossEvaluator;
        this.validationBatchLossEvaluator = builder.validationBatchLossEvaluator;
        this.trainEpochLossEvaluator = builder.trainEpochLossEvaluator;
        this.validationEpochLossEvaluator = builder.validationEpochLossEvaluator;
        this.earlyStoppingPatience = builder.earlyStoppingPatience;
        this.earlyStoppingMinDelta = builder.earlyStoppingMinDelta;
        this.resumeFromCheckpoint = builder.resumeFromCheckpoint;
        this.failOnCheckpointLoadError = builder.failOnCheckpointLoadError;
        this.checkpointStateFile = resolveCheckpointStateFile(config.checkpointDir());
        this.checkpointLoadReport = resumeFromCheckpoint
                ? loadCheckpointState(checkpointStateFile, config.epochs())
                : CheckpointLoadReport.notRequested();
        this.initialCheckpointState = checkpointLoadReport.state();
        if (checkpointLoadReport.failed() && failOnCheckpointLoadError) {
            String message = "Failed to resume canonical trainer checkpoint at "
                    + checkpointStateFile + ": " + checkpointLoadReport.error();
            throw new IllegalStateException(message);
        }
        if (initialCheckpointState != null) {
            this.globalStep = initialCheckpointState.globalStep();
            this.currentEpoch = Math.max(0, initialCheckpointState.nextEpoch() - 1);
        }
        this.summary = new TrainingSummary(
                0,
                Double.NaN,
                -1,
                null,
                null,
                0L,
                Map.of("runtime", TrainerRuntimeMode.CANONICAL_FALLBACK.value()));
    }

    public static Builder builder() {
        return new Builder();
    }

    public void fit(Iterable<?> trainLoader) {
        fit(trainLoader, null);
    }

    public void fit(Iterable<?> trainLoader, Iterable<?> validationLoader) {
        long startedAt = System.currentTimeMillis();
        listenerErrorCount = 0;
        notifyListenersSafely(listener -> listener.onTrainingStart(this));

        boolean resumedFromCheckpoint = initialCheckpointState != null;
        int startEpoch = resumedFromCheckpoint
                ? clamp(initialCheckpointState.nextEpoch(), 0, config.epochs())
                : 0;
        int completedEpochs = resumedFromCheckpoint
                ? clamp(initialCheckpointState.completedEpochs(), 0, config.epochs())
                : 0;
        if (completedEpochs < startEpoch) {
            completedEpochs = startEpoch;
        }

        int bestEpoch = resumedFromCheckpoint ? initialCheckpointState.bestValidationEpoch() : -1;
        double bestValLoss = resumedFromCheckpoint ? initialCheckpointState.bestValidationLoss() : Double.NaN;
        Double latestTrainLoss = resumedFromCheckpoint ? initialCheckpointState.latestTrainLoss() : null;
        Double latestValidationLoss = resumedFromCheckpoint ? initialCheckpointState.latestValidationLoss() : null;
        LossExecutionState lossState = new LossExecutionState();
        int epochsWithoutImprovement = resumedFromCheckpoint
                ? initialCheckpointState.epochsWithoutImprovement()
                : 0;
        boolean earlyStoppingTriggered = false;
        int stopEpoch = -1;
        String stopReason = "completed";
        String checkpointError = null;
        RuntimeException fatalError = null;
        try {
            for (int epoch = startEpoch; epoch < config.epochs() && !stopped.get(); epoch++) {
                currentEpoch = epoch;
                notifyEpochStart(epoch);

                latestTrainLoss = computeTrainLossForEpoch(epoch, trainLoader, lossState);
                notifyEpochEnd(epoch, latestTrainLoss);

                if (validationLoader != null) {
                    latestValidationLoss = computeValidationLossForEpoch(epoch, validationLoader, lossState);
                    notifyValidationEnd(epoch, latestValidationLoss);
                    boolean improved = Double.isNaN(bestValLoss)
                            || latestValidationLoss < bestValLoss - earlyStoppingMinDelta;
                    if (improved) {
                        bestValLoss = latestValidationLoss;
                        bestEpoch = epoch;
                        epochsWithoutImprovement = 0;
                    } else if (earlyStoppingPatience > 0) {
                        epochsWithoutImprovement++;
                    }
                }

                completedEpochs++;
                checkpointError = persistCheckpointSafely(
                        checkpointError,
                        epoch + 1,
                        completedEpochs,
                        bestEpoch,
                        bestValLoss,
                        latestTrainLoss,
                        latestValidationLoss,
                        epochsWithoutImprovement);
                if (earlyStoppingPatience > 0
                        && validationLoader != null
                        && epochsWithoutImprovement >= earlyStoppingPatience) {
                    earlyStoppingTriggered = true;
                    stopEpoch = epoch;
                    stopReason = "early-stopping";
                    stopped.set(true);
                    break;
                }
            }

            if (stopped.get()) {
                if (!earlyStoppingTriggered) {
                    stopEpoch = currentEpoch;
                    stopReason = "manual-stop";
                }
                int earlyStoppingEpoch = stopEpoch >= 0 ? stopEpoch : currentEpoch;
                notifyListenersSafely(listener -> listener.onEarlyStopping(this, earlyStoppingEpoch));
            }
        } catch (RuntimeException error) {
            fatalError = error;
            stopReason = "failed";
            notifyTrainingErrorSafely(error);
        }

        if (fatalError == null) {
            int nextEpoch = stopEpoch >= 0
                    ? clamp(stopEpoch + 1, 0, config.epochs())
                    : clamp(completedEpochs, 0, config.epochs());
            checkpointError = persistCheckpointSafely(
                    checkpointError,
                    nextEpoch,
                    completedEpochs,
                    bestEpoch,
                    bestValLoss,
                    latestTrainLoss,
                    latestValidationLoss,
                    epochsWithoutImprovement);
        }

        long durationMs = Math.max(0L, System.currentTimeMillis() - startedAt);
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("runtime", TrainerRuntimeMode.CANONICAL_FALLBACK.value());
        metadata.put("stopped", stopped.get());
        metadata.put("globalStep", globalStep);
        metadata.put("listenerErrors", listenerErrorCount);
        metadata.put("trainBatchLossHook", lossState.trainBatchHookUsed);
        metadata.put("trainEpochLossHook", lossState.trainEpochHookUsed);
        metadata.put("validationBatchLossHook", lossState.validationBatchHookUsed);
        metadata.put("validationEpochLossHook", lossState.validationEpochHookUsed);
        metadata.put("trainSyntheticFallbackUsed", lossState.trainSyntheticFallbackUsed);
        metadata.put("validationSyntheticFallbackUsed", lossState.validationSyntheticFallbackUsed);
        metadata.put("earlyStoppingPatience", earlyStoppingPatience);
        metadata.put("earlyStoppingMinDelta", earlyStoppingMinDelta);
        metadata.put("earlyStoppingTriggered", earlyStoppingTriggered);
        metadata.put("stopReason", stopReason);
        metadata.put("checkpointEnabled", checkpointStateFile != null);
        metadata.put("checkpointFormatVersion", CHECKPOINT_FORMAT_VERSION);
        metadata.put("checkpointResumeRequested", resumeFromCheckpoint);
        metadata.put("resumedFromCheckpoint", resumedFromCheckpoint);
        metadata.put("checkpointLoadFailed", checkpointLoadReport.failed());
        metadata.put("checkpointMigratedFromLegacy", checkpointLoadReport.migratedFromLegacy());
        if (checkpointLoadReport.detectedFormatVersion() != null) {
            metadata.put("checkpointDetectedFormatVersion", checkpointLoadReport.detectedFormatVersion());
        }
        if (checkpointLoadReport.error() != null) {
            metadata.put("checkpointLoadError", checkpointLoadReport.error());
        }
        if (checkpointStateFile != null) {
            metadata.put("checkpointFile", checkpointStateFile.toString());
        }
        if (resumedFromCheckpoint) {
            metadata.put("resumeStartEpoch", startEpoch);
            metadata.put("resumeGlobalStep", initialCheckpointState.globalStep());
        }
        if (checkpointError != null) {
            metadata.put("checkpointSaveFailed", true);
            metadata.put("checkpointError", checkpointError);
        } else {
            metadata.put("checkpointSaveFailed", false);
        }
        if (stopEpoch >= 0) {
            metadata.put("stopEpoch", stopEpoch);
        }
        if (fatalError != null) {
            metadata.put("failed", true);
            metadata.put("errorType", fatalError.getClass().getSimpleName());
        }
        summary = new TrainingSummary(
                completedEpochs,
                bestValLoss,
                bestEpoch,
                latestTrainLoss,
                latestValidationLoss,
                durationMs,
                Map.copyOf(metadata));

        notifyListenersSafely(listener -> listener.onTrainingEnd(this, summary));
        if (fatalError != null) {
            throw fatalError;
        }
    }

    private static double syntheticLoss(int seed) {
        return 1.0 / (1.0 + Math.max(seed, 1));
    }

    private static Path resolveCheckpointStateFile(Path checkpointDir) {
        if (checkpointDir == null) {
            return null;
        }
        return checkpointDir.resolve(CHECKPOINT_FILE_NAME);
    }

    private String persistCheckpointSafely(
            String existingError,
            int nextEpoch,
            int completedEpochs,
            int bestValidationEpoch,
            double bestValidationLoss,
            Double latestTrainLoss,
            Double latestValidationLoss,
            int epochsWithoutImprovement) {
        if (existingError != null || checkpointStateFile == null) {
            return existingError;
        }
        try {
            saveCheckpointState(
                    checkpointStateFile,
                    new CheckpointState(
                            clamp(nextEpoch, 0, config.epochs()),
                            clamp(completedEpochs, 0, config.epochs()),
                            Math.max(0, globalStep),
                            bestValidationEpoch,
                            bestValidationLoss,
                            latestTrainLoss,
                            latestValidationLoss,
                            Math.max(0, epochsWithoutImprovement)));
            return null;
        } catch (IOException ioe) {
            notifyTrainingErrorSafely(ioe);
            return ioe.getMessage();
        }
    }

    private static void saveCheckpointState(Path checkpointFile, CheckpointState state) throws IOException {
        if (checkpointFile == null) {
            return;
        }
        Path parent = checkpointFile.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        Properties properties = new Properties();
        properties.setProperty(CHECKPOINT_FORMAT_VERSION_KEY, Integer.toString(CHECKPOINT_FORMAT_VERSION));
        properties.setProperty("nextEpoch", Integer.toString(state.nextEpoch()));
        properties.setProperty("completedEpochs", Integer.toString(state.completedEpochs()));
        properties.setProperty("globalStep", Integer.toString(state.globalStep()));
        properties.setProperty("bestValidationEpoch", Integer.toString(state.bestValidationEpoch()));
        properties.setProperty("bestValidationLoss", Double.toString(state.bestValidationLoss()));
        properties.setProperty("latestTrainLoss",
                state.latestTrainLoss() == null ? "null" : Double.toString(state.latestTrainLoss()));
        properties.setProperty("latestValidationLoss",
                state.latestValidationLoss() == null ? "null" : Double.toString(state.latestValidationLoss()));
        properties.setProperty("epochsWithoutImprovement", Integer.toString(state.epochsWithoutImprovement()));

        try (Writer writer = Files.newBufferedWriter(
                checkpointFile,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE)) {
            properties.store(writer, "Gollek canonical trainer runtime checkpoint");
        }
    }

    private static CheckpointLoadReport loadCheckpointState(Path checkpointFile, int maxEpochs) {
        if (checkpointFile == null || !Files.isRegularFile(checkpointFile)) {
            return CheckpointLoadReport.notFound();
        }

        Properties properties = new Properties();
        try (Reader reader = Files.newBufferedReader(checkpointFile, StandardCharsets.UTF_8)) {
            properties.load(reader);
        } catch (IOException ioe) {
            return CheckpointLoadReport.failed("io-error: " + ioe.getMessage(), null);
        }

        String rawVersion = trimToNull(properties.getProperty(CHECKPOINT_FORMAT_VERSION_KEY));
        if (rawVersion == null) {
            CheckpointState migrated = decodeCheckpointState(properties, maxEpochs);
            if (migrated == null) {
                return CheckpointLoadReport.failed(
                        "legacy-checkpoint-missing-or-invalid-required-fields",
                        LEGACY_CHECKPOINT_FORMAT);
            }
            return CheckpointLoadReport.loaded(migrated, LEGACY_CHECKPOINT_FORMAT, true);
        }

        int parsedVersion;
        try {
            parsedVersion = Integer.parseInt(rawVersion);
        } catch (NumberFormatException nfe) {
            return CheckpointLoadReport.failed("invalid-format-version: " + rawVersion, rawVersion);
        }

        if (parsedVersion != CHECKPOINT_FORMAT_VERSION) {
            return CheckpointLoadReport.failed(
                    "unsupported-format-version: " + parsedVersion + " (supported: " + CHECKPOINT_FORMAT_VERSION + ")",
                    rawVersion);
        }

        CheckpointState decoded = decodeCheckpointState(properties, maxEpochs);
        if (decoded == null) {
            return CheckpointLoadReport.failed("checkpoint-payload-invalid", rawVersion);
        }
        return CheckpointLoadReport.loaded(decoded, rawVersion, false);
    }

    private static CheckpointState decodeCheckpointState(Properties properties, int maxEpochs) {
        Integer nextEpochRequired = parseRequiredInt(properties, "nextEpoch");
        Integer completedEpochsRequired = parseRequiredInt(properties, "completedEpochs");
        Integer globalStepRequired = parseRequiredInt(properties, "globalStep");
        if (nextEpochRequired == null || completedEpochsRequired == null || globalStepRequired == null) {
            return null;
        }

        int nextEpoch = clamp(nextEpochRequired, 0, maxEpochs);
        int completedEpochs = clamp(completedEpochsRequired, 0, maxEpochs);
        if (completedEpochs < nextEpoch) {
            completedEpochs = nextEpoch;
        }
        int globalStep = Math.max(0, globalStepRequired);
        int bestValidationEpoch = parseInt(properties, "bestValidationEpoch", -1);
        double bestValidationLoss = parseDouble(properties, "bestValidationLoss", Double.NaN);
        Double latestTrainLoss = parseNullableDouble(properties, "latestTrainLoss");
        Double latestValidationLoss = parseNullableDouble(properties, "latestValidationLoss");
        int epochsWithoutImprovement = Math.max(0, parseInt(properties, "epochsWithoutImprovement", 0));
        return new CheckpointState(
                nextEpoch,
                completedEpochs,
                globalStep,
                bestValidationEpoch,
                bestValidationLoss,
                latestTrainLoss,
                latestValidationLoss,
                epochsWithoutImprovement);
    }

    private static Integer parseRequiredInt(Properties properties, String key) {
        String value = trimToNull(properties.getProperty(key));
        if (value == null) {
            return null;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException nfe) {
            return null;
        }
    }

    private static int parseInt(Properties properties, String key, int fallback) {
        String value = properties.getProperty(key);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException nfe) {
            return fallback;
        }
    }

    private static double parseDouble(Properties properties, String key, double fallback) {
        String value = properties.getProperty(key);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException nfe) {
            return fallback;
        }
    }

    private static Double parseNullableDouble(Properties properties, String key) {
        String value = trimToNull(properties.getProperty(key));
        if (value == null || "null".equalsIgnoreCase(value)) {
            return null;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException nfe) {
            return null;
        }
    }

    private static String trimToNull(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(value, max));
    }

    private double computeTrainLossForEpoch(int epoch, Iterable<?> trainLoader, LossExecutionState state) {
        if (trainLoader == null) {
            return resolveEpochLoss(trainEpochLossEvaluator, epoch, null, true, state);
        }

        double totalLoss = 0.0;
        int batchCount = 0;
        for (Object batch : trainLoader) {
            if (stopped.get()) {
                break;
            }
            int step = globalStep++;
            notifyListenersSafely(listener -> listener.onBatchStart(this, step));
            double lossValue = computeTrainBatchLoss(epoch, step, batch, state);
            totalLoss += lossValue;
            batchCount++;
            notifyListenersSafely(listener -> listener.onBatchEnd(this, step, lossValue));
        }

        if (batchCount > 0) {
            return totalLoss / batchCount;
        }
        return resolveEpochLoss(trainEpochLossEvaluator, epoch, trainLoader, true, state);
    }

    private double computeValidationLossForEpoch(int epoch, Iterable<?> validationLoader, LossExecutionState state) {
        if (validationLoader == null) {
            return resolveEpochLoss(validationEpochLossEvaluator, epoch, null, false, state);
        }

        double totalLoss = 0.0;
        int batchCount = 0;
        for (Object batch : validationLoader) {
            if (stopped.get()) {
                break;
            }
            double lossValue = computeValidationBatchLoss(epoch, batchCount, batch, state);
            totalLoss += lossValue;
            batchCount++;
        }

        if (batchCount > 0) {
            return totalLoss / batchCount;
        }
        return resolveEpochLoss(validationEpochLossEvaluator, epoch, validationLoader, false, state);
    }

    private double computeTrainBatchLoss(int epoch, int step, Object batch, LossExecutionState state) {
        if (trainBatchLossEvaluator != null) {
            state.trainBatchHookUsed = true;
            return requireFinite(
                    trainBatchLossEvaluator.compute(this, epoch, step, batch),
                    "train batch loss");
        }
        state.trainSyntheticFallbackUsed = true;
        return syntheticLoss(step + 1);
    }

    private double computeValidationBatchLoss(int epoch, int validationStep, Object batch, LossExecutionState state) {
        if (validationBatchLossEvaluator != null) {
            state.validationBatchHookUsed = true;
            return requireFinite(
                    validationBatchLossEvaluator.compute(this, epoch, validationStep, batch),
                    "validation batch loss");
        }
        state.validationSyntheticFallbackUsed = true;
        return syntheticLoss((epoch + 1) * (validationStep + 1));
    }

    private double resolveEpochLoss(
            EpochLossEvaluator evaluator,
            int epoch,
            Iterable<?> loader,
            boolean trainPhase,
            LossExecutionState state) {
        if (evaluator != null) {
            if (trainPhase) {
                state.trainEpochHookUsed = true;
            } else {
                state.validationEpochHookUsed = true;
            }
            double loss = evaluator.compute(this, epoch, loader);
            return requireFinite(loss, trainPhase ? "train epoch loss" : "validation epoch loss");
        }

        if (trainPhase) {
            state.trainSyntheticFallbackUsed = true;
            return syntheticLoss(epoch);
        }

        state.validationSyntheticFallbackUsed = true;
        return syntheticLoss(epoch + 1);
    }

    private static double requireFinite(double loss, String label) {
        if (!Double.isFinite(loss)) {
            throw new IllegalArgumentException(label + " must be finite, got " + loss);
        }
        return loss;
    }

    private void notifyEpochStart(int epoch) {
        notifyListenersSafely(listener -> listener.onEpochStart(this, epoch));
    }

    private void notifyEpochEnd(int epoch, double trainLoss) {
        notifyListenersSafely(listener -> listener.onEpochEnd(this, epoch, trainLoss));
    }

    private void notifyValidationEnd(int epoch, double valLoss) {
        notifyListenersSafely(listener -> listener.onValidationEnd(this, epoch, valLoss));
    }

    private void notifyListenersSafely(ListenerAction action) {
        for (TrainingListener listener : listeners) {
            try {
                action.invoke(listener);
            } catch (RuntimeException callbackError) {
                listenerErrorCount++;
                notifyTrainingErrorSafely(callbackError);
            }
        }
    }

    private void notifyTrainingErrorSafely(Exception error) {
        for (TrainingListener listener : listeners) {
            try {
                listener.onTrainingError(this, error);
            } catch (RuntimeException ignored) {
                // Listener error handlers must never crash trainer lifecycle.
            }
        }
    }

    @FunctionalInterface
    private interface ListenerAction {
        void invoke(TrainingListener listener);
    }

    @FunctionalInterface
    public interface BatchLossEvaluator {
        double compute(TrainerSession session, int epoch, int step, Object batch);
    }

    @FunctionalInterface
    public interface EpochLossEvaluator {
        double compute(TrainerSession session, int epoch, Iterable<?> loader);
    }

    private static final class LossExecutionState {
        private boolean trainBatchHookUsed;
        private boolean trainEpochHookUsed;
        private boolean validationBatchHookUsed;
        private boolean validationEpochHookUsed;
        private boolean trainSyntheticFallbackUsed;
        private boolean validationSyntheticFallbackUsed;
    }

    private record CheckpointState(
            int nextEpoch,
            int completedEpochs,
            int globalStep,
            int bestValidationEpoch,
            double bestValidationLoss,
            Double latestTrainLoss,
            Double latestValidationLoss,
            int epochsWithoutImprovement) {
    }

    private record CheckpointLoadReport(
            CheckpointState state,
            boolean failed,
            boolean migratedFromLegacy,
            String detectedFormatVersion,
            String error) {
        private static CheckpointLoadReport notRequested() {
            return new CheckpointLoadReport(null, false, false, null, null);
        }

        private static CheckpointLoadReport notFound() {
            return new CheckpointLoadReport(null, false, false, null, null);
        }

        private static CheckpointLoadReport loaded(
                CheckpointState state,
                String detectedFormatVersion,
                boolean migratedFromLegacy) {
            return new CheckpointLoadReport(state, false, migratedFromLegacy, detectedFormatVersion, null);
        }

        private static CheckpointLoadReport failed(String error, String detectedFormatVersion) {
            return new CheckpointLoadReport(null, true, false, detectedFormatVersion, error);
        }
    }

    @Override
    public int currentEpoch() {
        return currentEpoch;
    }

    @Override
    public int globalStep() {
        return globalStep;
    }

    @Override
    public TrainerConfig config() {
        return config;
    }

    @Override
    public TrainingSummary summary() {
        return summary;
    }

    @Override
    public boolean isStopped() {
        return stopped.get();
    }

    @Override
    public void stop() {
        stopped.set(true);
    }

    @Override
    public void close() {
        stop();
        for (TrainingListener listener : listeners) {
            try {
                listener.close();
            } catch (RuntimeException ignored) {
                // Close is best-effort for listener integrations.
            }
        }
    }

    public static final class Builder {
        private int epochs = 1;
        private double gradientClip = 0.0;
        private boolean mixedPrecision = false;
        private Path checkpointDir = null;
        private final List<TrainingListener> listeners = new ArrayList<>();
        private BatchLossEvaluator trainBatchLossEvaluator;
        private BatchLossEvaluator validationBatchLossEvaluator;
        private EpochLossEvaluator trainEpochLossEvaluator;
        private EpochLossEvaluator validationEpochLossEvaluator;
        private int earlyStoppingPatience = 0;
        private double earlyStoppingMinDelta = 0.0;
        private boolean resumeFromCheckpoint = false;
        private boolean failOnCheckpointLoadError = true;

        private Builder() {
        }

        // Compatibility no-ops while migration is in progress.
        public Builder model(Object ignored) {
            return this;
        }

        public Builder optimizer(Object ignored) {
            return this;
        }

        public Builder loss(Object candidate) {
            if (candidate instanceof BatchLossEvaluator batchLossEvaluator) {
                this.trainBatchLossEvaluator = batchLossEvaluator;
                this.validationBatchLossEvaluator = batchLossEvaluator;
                return this;
            }
            if (candidate instanceof EpochLossEvaluator epochLossEvaluator) {
                this.trainEpochLossEvaluator = epochLossEvaluator;
                this.validationEpochLossEvaluator = epochLossEvaluator;
                return this;
            }
            if (candidate instanceof ToDoubleFunction<?> toDoubleFunction) {
                ToDoubleFunction<Object> adapted = unsafeToDoubleFunction(toDoubleFunction);
                this.trainBatchLossEvaluator = (session, epoch, step, batch) -> adapted.applyAsDouble(batch);
                this.validationBatchLossEvaluator = (session, epoch, step, batch) -> adapted.applyAsDouble(batch);
                return this;
            }
            if (candidate instanceof ToDoubleBiFunction<?, ?> toDoubleBiFunction) {
                ToDoubleBiFunction<TrainerSession, Object> adapted = unsafeToDoubleBiFunction(toDoubleBiFunction);
                this.trainBatchLossEvaluator = (session, epoch, step, batch) -> adapted.applyAsDouble(session, batch);
                this.validationBatchLossEvaluator = (session, epoch, step, batch) -> adapted.applyAsDouble(session, batch);
                return this;
            }
            return this;
        }

        public Builder scheduler(Object ignored) {
            return this;
        }

        public Builder trainBatchLoss(BatchLossEvaluator evaluator) {
            this.trainBatchLossEvaluator = evaluator;
            return this;
        }

        public Builder validationBatchLoss(BatchLossEvaluator evaluator) {
            this.validationBatchLossEvaluator = evaluator;
            return this;
        }

        public Builder trainEpochLoss(EpochLossEvaluator evaluator) {
            this.trainEpochLossEvaluator = evaluator;
            return this;
        }

        public Builder validationEpochLoss(EpochLossEvaluator evaluator) {
            this.validationEpochLossEvaluator = evaluator;
            return this;
        }

        public Builder earlyStopping(int patience) {
            this.earlyStoppingPatience = Math.max(0, patience);
            this.earlyStoppingMinDelta = Math.max(0.0, this.earlyStoppingMinDelta);
            return this;
        }

        public Builder earlyStopping(int patience, double minDelta) {
            this.earlyStoppingPatience = Math.max(0, patience);
            this.earlyStoppingMinDelta = Math.max(0.0, minDelta);
            return this;
        }

        public Builder callback(Object callback) {
            if (callback instanceof TrainingListener listener && listener != null) {
                listeners.add(listener);
            }
            return this;
        }

        public Builder callbacks(List<?> callbacks) {
            if (callbacks == null) {
                return this;
            }
            for (Object callback : callbacks) {
                callback(callback);
            }
            return this;
        }

        public Builder listener(TrainingListener listener) {
            if (listener == null) {
                return this;
            }
            listeners.add(listener);
            return this;
        }

        public Builder listeners(List<? extends TrainingListener> listeners) {
            if (listeners == null) {
                return this;
            }
            for (TrainingListener listener : listeners) {
                if (listener != null) {
                    this.listeners.add(listener);
                }
            }
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = Math.max(1, epochs);
            return this;
        }

        public Builder gradientClip(double gradientClip) {
            this.gradientClip = Math.max(0.0, gradientClip);
            return this;
        }

        public Builder mixedPrecision(boolean mixedPrecision) {
            this.mixedPrecision = mixedPrecision;
            return this;
        }

        public Builder checkpointDir(Path checkpointDir) {
            this.checkpointDir = checkpointDir;
            return this;
        }

        public Builder resumeFromCheckpoint() {
            return resumeFromCheckpoint(true);
        }

        public Builder resumeFromCheckpoint(boolean resumeFromCheckpoint) {
            this.resumeFromCheckpoint = resumeFromCheckpoint;
            return this;
        }

        public Builder failOnCheckpointLoadError(boolean failOnCheckpointLoadError) {
            this.failOnCheckpointLoadError = failOnCheckpointLoadError;
            return this;
        }

        public CanonicalTrainerRuntime build() {
            return new CanonicalTrainerRuntime(this);
        }

        @SuppressWarnings("unchecked")
        private static ToDoubleFunction<Object> unsafeToDoubleFunction(ToDoubleFunction<?> function) {
            return (ToDoubleFunction<Object>) function;
        }

        @SuppressWarnings("unchecked")
        private static ToDoubleBiFunction<TrainerSession, Object> unsafeToDoubleBiFunction(
                ToDoubleBiFunction<?, ?> function) {
            return (ToDoubleBiFunction<TrainerSession, Object>) function;
        }
    }
}
