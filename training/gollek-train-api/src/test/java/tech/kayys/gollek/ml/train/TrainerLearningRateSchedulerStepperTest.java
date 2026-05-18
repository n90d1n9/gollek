package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.optim.LRScheduler;
import tech.kayys.gollek.ml.optim.Optimizer;
import tech.kayys.gollek.ml.optim.SGD;

class TrainerLearningRateSchedulerStepperTest {

    @Test
    void stepsOnlyWhenConfiguredUnitMatches() {
        RecordingScheduler scheduler = new RecordingScheduler();
        TrainerLearningRateSchedulerStepper stepper = new TrainerLearningRateSchedulerStepper(
                scheduler,
                CanonicalTrainer.SchedulerStepUnit.EPOCH);

        assertFalse(stepper.step(CanonicalTrainer.SchedulerStepUnit.BATCH));
        assertEquals(0, scheduler.plainSteps);
        assertEquals(0, stepper.stepCount());

        assertTrue(stepper.step(CanonicalTrainer.SchedulerStepUnit.EPOCH));
        assertEquals(1, scheduler.plainSteps);
        assertEquals(1, stepper.stepCount());
    }

    @Test
    void validationStepPassesMonitorValueToScheduler() {
        RecordingScheduler scheduler = new RecordingScheduler();
        TrainerLearningRateSchedulerStepper stepper = new TrainerLearningRateSchedulerStepper(
                scheduler,
                CanonicalTrainer.SchedulerStepUnit.VALIDATION);

        assertTrue(stepper.step(CanonicalTrainer.SchedulerStepUnit.VALIDATION, 0.42));

        assertEquals(0, scheduler.plainSteps);
        assertEquals(1, scheduler.metricSteps);
        assertEquals(0.42, scheduler.lastMetric, 1e-12);
        assertEquals(1, stepper.stepCount());
    }

    @Test
    void disabledSchedulerDoesNotStepAndPublishesEmptyState() {
        TrainerLearningRateSchedulerStepper stepper = new TrainerLearningRateSchedulerStepper(
                null,
                CanonicalTrainer.SchedulerStepUnit.BATCH);

        assertFalse(stepper.enabled());
        assertFalse(stepper.supportsStateDict());
        assertFalse(stepper.step(CanonicalTrainer.SchedulerStepUnit.BATCH));
        assertEquals("none", stepper.schedulerType());
        assertEquals(Map.of(), stepper.stateSnapshot());
        assertEquals(0, stepper.stepCount());
    }

    @Test
    void restoresCheckpointStepCountAndSnapshotsSchedulerState() {
        RecordingScheduler scheduler = new RecordingScheduler();
        scheduler.supportState = true;
        scheduler.state = Map.of("stepCount", 7, "currentLr", 0.025);
        TrainerLearningRateSchedulerStepper stepper = new TrainerLearningRateSchedulerStepper(
                scheduler,
                CanonicalTrainer.SchedulerStepUnit.BATCH);

        stepper.restoreStepCount(-3);
        assertEquals(0, stepper.stepCount());
        stepper.restoreStepCount(7);

        assertTrue(stepper.enabled());
        assertTrue(stepper.supportsStateDict());
        assertEquals("RecordingScheduler", stepper.schedulerType());
        assertEquals(7, stepper.stepCount());
        assertEquals(Map.of("stepCount", 7, "currentLr", 0.025), stepper.stateSnapshot());
    }

    private static final class RecordingScheduler extends LRScheduler {
        private int plainSteps;
        private int metricSteps;
        private double lastMetric = Double.NaN;
        private boolean supportState;
        private Map<String, Object> state = Map.of();

        private RecordingScheduler() {
            super(optimizer());
        }

        @Override
        public void step() {
            plainSteps++;
        }

        @Override
        public void step(double metric) {
            metricSteps++;
            lastMetric = metric;
        }

        @Override
        public float getLr() {
            return optimizer.learningRate();
        }

        @Override
        public boolean supportsStateDict() {
            return supportState;
        }

        @Override
        public Map<String, Object> stateDict() {
            return state;
        }

        private static Optimizer optimizer() {
            Parameter parameter = new Parameter(GradTensor.of(new float[] {1f}, 1));
            return SGD.builder(List.of(parameter), 0.1f).build();
        }
    }
}
