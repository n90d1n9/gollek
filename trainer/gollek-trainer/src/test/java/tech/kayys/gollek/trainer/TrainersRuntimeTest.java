package tech.kayys.gollek.trainer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.ToDoubleFunction;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.trainer.api.TrainerRuntimeMode;
import tech.kayys.gollek.trainer.api.TrainerSession;
import tech.kayys.gollek.trainer.api.TrainingListener;
import tech.kayys.gollek.trainer.api.TrainingSummary;

class TrainersRuntimeTest {

    @Test
    void runtimeIsAvailableFromCanonicalEntryPoint() {
        assertTrue(Trainers.runtimeAvailable());
    }

    @Test
    void runtimeModeIsExplicit() {
        assertTrue(Set.of("legacy-bridge", "canonical-fallback").contains(Trainers.runtimeMode()));
        assertTrue(Set.of(TrainerRuntimeMode.LEGACY_BRIDGE, TrainerRuntimeMode.CANONICAL_FALLBACK)
                .contains(Trainers.runtimeModeType()));
        assertEquals(Trainers.runtimeModeType().value(), Trainers.runtimeMode());
    }

    @Test
    void builderCreatesATrainerSession() throws Exception {
        Object builder = Trainers.builder();
        assertNotNull(builder);

        Method buildMethod = builder.getClass().getMethod("build");
        Object runtime = buildMethod.invoke(builder);
        assertTrue(runtime instanceof TrainerSession);
    }

    @Test
    void canonicalFallbackCanRunSyntheticFitLoop() throws Exception {
        Object builder = Trainers.builder();
        Method epochs = builder.getClass().getMethod("epochs", int.class);
        epochs.invoke(builder, 2);

        Method buildMethod = builder.getClass().getMethod("build");
        Object runtime = buildMethod.invoke(builder);

        if (runtime instanceof CanonicalTrainerRuntime canonical) {
            canonical.fit(List.of("a", "b", "c"), List.of("v"));
            assertEquals(2, canonical.summary().epochCount());
            assertTrue(canonical.globalStep() > 0);
            return;
        }

        // If a richer legacy runtime is available, just assert session contract.
        assertTrue(runtime instanceof TrainerSession);
    }

    @Test
    void typedSessionHelpersWork() {
        TrainerSession session = Trainers.newSession(3);
        assertNotNull(session);
        assertEquals(3, session.config().epochs());

        if (session instanceof CanonicalTrainerRuntime canonical) {
            canonical.fit(List.of("a", "b"), List.of("v"));
            TrainingSummary summary = canonical.summary();
            assertEquals(3, summary.epochCount());
            assertTrue(summary.durationMs() >= 0);
        }
    }

    @Test
    void sessionBuilderFallsBackToCanonicalWhenLegacyInputsAreMissing() {
        TrainerSession session = Trainers.sessionBuilder()
                .epochs(2)
                .gradientClip(0.5)
                .mixedPrecision(true)
                .build();

        assertNotNull(session);
        assertTrue(session instanceof CanonicalTrainerRuntime);
        assertEquals(2, session.config().epochs());
        assertEquals(0.5, session.config().gradientClip(), 1e-9);
        assertTrue(session.config().mixedPrecision());
    }

    @Test
    void canonicalFallbackNotifiesLifecycleListeners() {
        AtomicInteger starts = new AtomicInteger();
        AtomicInteger epochStarts = new AtomicInteger();
        AtomicInteger epochEnds = new AtomicInteger();
        AtomicInteger batchEnds = new AtomicInteger();
        AtomicInteger trainingEnds = new AtomicInteger();

        TrainingListener listener = new TrainingListener() {
            @Override
            public void onTrainingStart(TrainerSession session) {
                starts.incrementAndGet();
            }

            @Override
            public void onEpochStart(TrainerSession session, int epoch) {
                epochStarts.incrementAndGet();
            }

            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                epochEnds.incrementAndGet();
            }

            @Override
            public void onBatchEnd(TrainerSession session, int step, double loss) {
                batchEnds.incrementAndGet();
            }

            @Override
            public void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
                trainingEnds.incrementAndGet();
            }
        };

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(2)
                .listener(listener)
                .build();

        runtime.fit(List.of("a", "b", "c"), List.of("v"));

        assertEquals(1, starts.get());
        assertEquals(2, epochStarts.get());
        assertEquals(2, epochEnds.get());
        assertTrue(batchEnds.get() >= 6);
        assertEquals(1, trainingEnds.get());
    }

    @Test
    void canonicalFallbackNotifiesTrainingErrorAndRethrows() {
        AtomicReference<Exception> capturedError = new AtomicReference<>();
        AtomicInteger trainingEnds = new AtomicInteger();

        TrainingListener listener = new TrainingListener() {
            @Override
            public void onTrainingError(TrainerSession session, Exception error) {
                capturedError.set(error);
            }

            @Override
            public void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
                trainingEnds.incrementAndGet();
            }
        };

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(2)
                .listener(listener)
                .build();

        Iterable<Object> brokenLoader = () -> new Iterator<>() {
            private boolean first = true;

            @Override
            public boolean hasNext() {
                return first;
            }

            @Override
            public Object next() {
                first = false;
                throw new IllegalStateException("synthetic loader failure");
            }
        };

        IllegalStateException thrown = assertThrows(IllegalStateException.class,
                () -> runtime.fit(brokenLoader));
        assertEquals("synthetic loader failure", thrown.getMessage());
        assertNotNull(capturedError.get());
        assertEquals("synthetic loader failure", capturedError.get().getMessage());
        assertEquals(1, trainingEnds.get());
    }

    @Test
    void nullListenersAreIgnoredByCompatibilityBuilders() {
        TrainerSession session = Trainers.sessionBuilder()
                .listener(null)
                .listeners(null)
                .epochs(1)
                .build();

        assertNotNull(session);
        assertTrue(session instanceof CanonicalTrainerRuntime);
    }

    @Test
    void listenerCallbackFailuresDoNotAbortCanonicalRuntime() {
        AtomicInteger errorEvents = new AtomicInteger();

        TrainingListener listener = new TrainingListener() {
            @Override
            public void onBatchEnd(TrainerSession session, int step, double loss) {
                throw new IllegalStateException("listener batch failure");
            }

            @Override
            public void onTrainingError(TrainerSession session, Exception error) {
                errorEvents.incrementAndGet();
            }
        };

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(2)
                .listener(listener)
                .build();

        runtime.fit(List.of("a", "b"), List.of("v"));

        assertEquals(2, runtime.summary().epochCount());
        assertTrue(((Number) runtime.summary().metadata().get("listenerErrors")).intValue() > 0);
        assertTrue(errorEvents.get() > 0);
    }

    @Test
    void canonicalFallbackSupportsCustomBatchLossHooks() {
        AtomicInteger trainBatches = new AtomicInteger();
        AtomicInteger valBatches = new AtomicInteger();

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(2)
                .trainBatchLoss((session, epoch, step, batch) -> {
                    trainBatches.incrementAndGet();
                    return ((Number) batch).doubleValue() + epoch;
                })
                .validationBatchLoss((session, epoch, step, batch) -> {
                    valBatches.incrementAndGet();
                    return ((Number) batch).doubleValue() + 10.0 + epoch;
                })
                .build();

        runtime.fit(List.of(1, 2, 3), List.of(4, 5));

        TrainingSummary summary = runtime.summary();
        assertEquals(2, summary.epochCount());
        assertEquals(3.0, summary.latestTrainLoss(), 1e-9);
        assertEquals(15.5, summary.latestValidationLoss(), 1e-9);
        assertEquals(6, trainBatches.get());
        assertEquals(4, valBatches.get());
        assertEquals(Boolean.FALSE, summary.metadata().get("trainSyntheticFallbackUsed"));
        assertEquals(Boolean.FALSE, summary.metadata().get("validationSyntheticFallbackUsed"));
    }

    @Test
    void canonicalFallbackSupportsEpochLossHooks() {
        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(2)
                .trainEpochLoss((session, epoch, loader) -> 100.0 - epoch)
                .validationEpochLoss((session, epoch, loader) -> 5.0 + epoch)
                .build();

        runtime.fit(null, List.of());

        TrainingSummary summary = runtime.summary();
        assertEquals(2, summary.epochCount());
        assertEquals(99.0, summary.latestTrainLoss(), 1e-9);
        assertEquals(6.0, summary.latestValidationLoss(), 1e-9);
        assertEquals(Boolean.FALSE, summary.metadata().get("trainSyntheticFallbackUsed"));
        assertEquals(Boolean.FALSE, summary.metadata().get("validationSyntheticFallbackUsed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("trainEpochLossHook"));
        assertEquals(Boolean.TRUE, summary.metadata().get("validationEpochLossHook"));
    }

    @Test
    void canonicalLossCompatibilityAcceptsToDoubleFunction() {
        ToDoubleFunction<Object> lossFn = batch -> ((Number) batch).doubleValue();

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(1)
                .loss(lossFn)
                .build();

        runtime.fit(List.of(2, 4, 6), null);
        assertEquals(4.0, runtime.summary().latestTrainLoss(), 1e-9);
        assertEquals(Boolean.FALSE, runtime.summary().metadata().get("trainSyntheticFallbackUsed"));
    }

    @Test
    void sessionBuilderFallbackCanUseCanonicalLossHook() {
        TrainerSession session = Trainers.sessionBuilder()
                .epochs(1)
                .loss((CanonicalTrainerRuntime.BatchLossEvaluator) (runtime, epoch, step, batch) -> ((Number) batch)
                        .doubleValue() * 2.0)
                .build();

        assertTrue(session instanceof CanonicalTrainerRuntime);
        CanonicalTrainerRuntime runtime = (CanonicalTrainerRuntime) session;
        runtime.fit(List.of(1, 2, 3), null);
        assertEquals(4.0, runtime.summary().latestTrainLoss(), 1e-9);
    }

    @Test
    void canonicalFallbackSupportsEarlyStoppingMetadataAndCallback() {
        AtomicInteger earlyStoppingEpoch = new AtomicInteger(-1);

        TrainingListener listener = new TrainingListener() {
            @Override
            public void onEarlyStopping(TrainerSession session, int epoch) {
                earlyStoppingEpoch.set(epoch);
            }
        };

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(10)
                .trainEpochLoss((session, epoch, loader) -> 10.0 - epoch)
                .validationEpochLoss((session, epoch, loader) -> 1.0)
                .earlyStopping(2)
                .listener(listener)
                .build();

        runtime.fit(null, List.of());

        TrainingSummary summary = runtime.summary();
        assertEquals(3, summary.epochCount());
        assertEquals(0, summary.bestValidationEpoch());
        assertEquals("early-stopping", summary.metadata().get("stopReason"));
        assertEquals(Boolean.TRUE, summary.metadata().get("earlyStoppingTriggered"));
        assertEquals(2, ((Number) summary.metadata().get("earlyStoppingPatience")).intValue());
        assertEquals(2, ((Number) summary.metadata().get("stopEpoch")).intValue());
        assertEquals(2, earlyStoppingEpoch.get());
    }

    @Test
    void earlyStoppingMinDeltaIsAppliedBeforeCountingPatience() {
        double[] validationLosses = {1.0, 0.95, 0.94, 0.93};

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(10)
                .trainEpochLoss((session, epoch, loader) -> 5.0)
                .validationEpochLoss((session, epoch, loader) ->
                        validationLosses[Math.min(epoch, validationLosses.length - 1)])
                .earlyStopping(1, 0.1)
                .build();

        runtime.fit(null, List.of());

        TrainingSummary summary = runtime.summary();
        assertEquals(2, summary.epochCount());
        assertEquals("early-stopping", summary.metadata().get("stopReason"));
        assertEquals(Boolean.TRUE, summary.metadata().get("earlyStoppingTriggered"));
        assertEquals(1, ((Number) summary.metadata().get("stopEpoch")).intValue());
    }

    @Test
    void canonicalFallbackCanResumeFromCheckpointState() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-trainer-checkpoint");

        TrainingListener stopAtSecondEpoch = new TrainingListener() {
            @Override
            public void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
                if (epoch == 1) {
                    session.stop();
                }
            }
        };

        CanonicalTrainerRuntime firstRun = Trainers.canonicalBuilder()
                .epochs(4)
                .checkpointDir(checkpointDir)
                .trainEpochLoss((session, epoch, loader) -> 100.0 - epoch)
                .validationEpochLoss((session, epoch, loader) -> 10.0 - epoch)
                .listener(stopAtSecondEpoch)
                .build();

        firstRun.fit(null, List.of("v"));
        assertEquals(2, firstRun.summary().epochCount());
        assertEquals("manual-stop", firstRun.summary().metadata().get("stopReason"));
        assertEquals(Boolean.FALSE, firstRun.summary().metadata().get("resumedFromCheckpoint"));

        CanonicalTrainerRuntime resumedRun = Trainers.canonicalBuilder()
                .epochs(4)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .trainEpochLoss((session, epoch, loader) -> 100.0 - epoch)
                .validationEpochLoss((session, epoch, loader) -> 10.0 - epoch)
                .build();

        resumedRun.fit(null, List.of("v"));
        TrainingSummary resumedSummary = resumedRun.summary();
        assertEquals(4, resumedSummary.epochCount());
        assertEquals(3, resumedSummary.bestValidationEpoch());
        assertEquals(Boolean.TRUE, resumedSummary.metadata().get("resumedFromCheckpoint"));
        assertEquals(2, ((Number) resumedSummary.metadata().get("resumeStartEpoch")).intValue());
        assertEquals(Boolean.FALSE, resumedSummary.metadata().get("checkpointSaveFailed"));
        assertEquals(checkpointDir.resolve("canonical-runtime.state").toString(),
                resumedSummary.metadata().get("checkpointFile"));
    }

    @Test
    void sessionBuilderResumeModeBuildsCanonicalRuntime() {
        TrainerSession session = Trainers.sessionBuilder()
                .epochs(1)
                .resumeFromCheckpoint()
                .build();

        assertTrue(session instanceof CanonicalTrainerRuntime);
    }

    @Test
    void resumeRejectsMissingRuntimeCheckpointByDefault() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-trainer-checkpoint-missing");

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> Trainers.canonicalBuilder()
                        .epochs(2)
                        .checkpointDir(checkpointDir)
                        .resumeFromCheckpoint()
                        .build());
        assertTrue(error.getMessage().contains("Missing canonical trainer checkpoint"));
    }

    @Test
    void resumeCanFallbackToFreshStateWhenRuntimeCheckpointIsMissingAndGuardIsDisabled() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-trainer-checkpoint-missing-lenient");

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .trainEpochLoss((session, epoch, loader) -> 3.0)
                .build();

        runtime.fit(null, null);
        TrainingSummary summary = runtime.summary();
        assertEquals(2, summary.epochCount());
        assertEquals(Boolean.FALSE, summary.metadata().get("resumedFromCheckpoint"));
        assertEquals(Boolean.TRUE, summary.metadata().get("checkpointMissingOnResume"));
        assertEquals(Boolean.FALSE, summary.metadata().get("checkpointLoadFailed"));
        assertEquals("checkpoint-not-found", summary.metadata().get("checkpointLoadError"));
        assertEquals(Boolean.TRUE, summary.metadata().get("checkpointPresent"));
    }

    @Test
    void resumeRejectsUnsupportedCheckpointFormatByDefault() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-trainer-checkpoint-bad-version");
        Files.writeString(checkpointDir.resolve("canonical-runtime.state"),
                String.join("\n",
                        "formatVersion=999",
                        "nextEpoch=1",
                        "completedEpochs=1",
                        "globalStep=2",
                        "bestValidationEpoch=0",
                        "bestValidationLoss=1.0"));

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> Trainers.canonicalBuilder()
                        .epochs(2)
                        .checkpointDir(checkpointDir)
                        .resumeFromCheckpoint()
                        .build());
        assertTrue(error.getMessage().contains("unsupported-format-version"));
    }

    @Test
    void resumeCanFallbackToFreshStateWhenCheckpointLoadGuardIsDisabled() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-trainer-checkpoint-lenient");
        Files.writeString(checkpointDir.resolve("canonical-runtime.state"),
                String.join("\n",
                        "formatVersion=999",
                        "nextEpoch=1",
                        "completedEpochs=1",
                        "globalStep=2"));

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(2)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .failOnCheckpointLoadError(false)
                .trainEpochLoss((session, epoch, loader) -> 3.0)
                .build();

        runtime.fit(null, null);
        TrainingSummary summary = runtime.summary();
        assertEquals(2, summary.epochCount());
        assertEquals(Boolean.TRUE, summary.metadata().get("checkpointLoadFailed"));
        assertEquals(Boolean.FALSE, summary.metadata().get("resumedFromCheckpoint"));
        assertEquals("999", summary.metadata().get("checkpointDetectedFormatVersion"));
        assertTrue(((String) summary.metadata().get("checkpointLoadError")).contains("unsupported-format-version"));
    }

    @Test
    void legacyCheckpointWithoutVersionIsMigratedOnResume() throws Exception {
        Path checkpointDir = Files.createTempDirectory("gollek-trainer-checkpoint-legacy");
        Files.writeString(checkpointDir.resolve("canonical-runtime.state"),
                String.join("\n",
                        "nextEpoch=1",
                        "completedEpochs=1",
                        "globalStep=3",
                        "bestValidationEpoch=0",
                        "bestValidationLoss=2.0",
                        "latestTrainLoss=2.5",
                        "latestValidationLoss=2.0",
                        "epochsWithoutImprovement=0"));

        CanonicalTrainerRuntime runtime = Trainers.canonicalBuilder()
                .epochs(3)
                .checkpointDir(checkpointDir)
                .resumeFromCheckpoint()
                .trainEpochLoss((session, epoch, loader) -> 10.0 - epoch)
                .validationEpochLoss((session, epoch, loader) -> 10.0 - epoch)
                .build();

        runtime.fit(null, List.of("v"));
        TrainingSummary summary = runtime.summary();
        assertEquals(3, summary.epochCount());
        assertEquals(Boolean.TRUE, summary.metadata().get("resumedFromCheckpoint"));
        assertEquals(Boolean.FALSE, summary.metadata().get("checkpointLoadFailed"));
        assertEquals(Boolean.TRUE, summary.metadata().get("checkpointMigratedFromLegacy"));
        assertEquals("legacy-properties", summary.metadata().get("checkpointDetectedFormatVersion"));
        assertEquals(1, ((Number) summary.metadata().get("resumeStartEpoch")).intValue());
    }
}
