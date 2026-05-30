package tech.kayys.gollek.ml.optim;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs multiple learning-rate schedulers in sequence.
 *
 * <p>Milestones define the global step at which the next scheduler becomes
 * active. For example, two schedulers with milestone {@code 2} will use the
 * first scheduler for composite steps 1 and 2, then use the second scheduler
 * from step 3 onward.</p>
 */
public final class SequentialLR extends LRScheduler {
    private final List<LRScheduler> schedulers;
    private final int[] milestones;
    private int currentStep;
    private int activeIndex;
    private float currentLr;

    public SequentialLR(Optimizer optimizer, List<LRScheduler> schedulers, int... milestones) {
        super(optimizer);
        if (schedulers == null || schedulers.isEmpty()) {
            throw new IllegalArgumentException("schedulers must contain at least one scheduler");
        }
        if (milestones == null) {
            throw new IllegalArgumentException("milestones must not be null");
        }
        if (milestones.length != schedulers.size() - 1) {
            throw new IllegalArgumentException(
                    "milestones length must be schedulers.size() - 1, got: "
                            + milestones.length + " for " + schedulers.size() + " schedulers");
        }
        this.schedulers = List.copyOf(validateSchedulers(optimizer, schedulers));
        this.milestones = validateMilestones(milestones);
        this.activeIndex = schedulerIndexForStep(0);
        this.currentLr = this.schedulers.get(activeIndex).getLr();
        setLearningRate(currentLr);
    }

    @Override
    public void step() {
        activeIndex = schedulerIndexForStep(currentStep);
        LRScheduler active = schedulers.get(activeIndex);
        active.step();
        currentLr = active.getLr();
        setLearningRate(currentLr);
        currentStep++;
    }

    @Override
    public void step(double metric) {
        activeIndex = schedulerIndexForStep(currentStep);
        LRScheduler active = schedulers.get(activeIndex);
        active.step(metric);
        currentLr = active.getLr();
        setLearningRate(currentLr);
        currentStep++;
    }

    @Override
    public float getLr() {
        return currentLr;
    }

    public int currentStep() {
        return currentStep;
    }

    public int activeIndex() {
        return activeIndex;
    }

    public LRScheduler activeScheduler() {
        return schedulers.get(activeIndex);
    }

    public List<LRScheduler> schedulers() {
        return schedulers;
    }

    public int[] milestones() {
        return Arrays.copyOf(milestones, milestones.length);
    }

    @Override
    public boolean supportsStateDict() {
        for (LRScheduler scheduler : schedulers) {
            if (!scheduler.supportsStateDict()) {
                return false;
            }
        }
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "SequentialLR");
        state.put("currentStep", currentStep);
        state.put("activeIndex", activeIndex);
        state.put("currentLr", currentLr);
        state.put("milestones", Arrays.copyOf(milestones, milestones.length));
        List<Map<String, Object>> childStates = new ArrayList<>(schedulers.size());
        for (LRScheduler scheduler : schedulers) {
            childStates.add(scheduler.stateDict());
        }
        state.put("schedulers", childStates);
        return state;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }
        Object schedulerName = state.get("scheduler");
        if (schedulerName instanceof String name && !"SequentialLR".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint scheduler mismatch: expected SequentialLR but got " + name);
        }
        requireMilestonesMatch(state.get("milestones"));
        Object rawSchedulers = state.get("schedulers");
        if (rawSchedulers != null) {
            if (!(rawSchedulers instanceof List<?> childStates)) {
                throw new IllegalArgumentException("Invalid SequentialLR checkpoint payload: schedulers must be a List");
            }
            if (childStates.size() != schedulers.size()) {
                throw new IllegalArgumentException(
                        "Invalid SequentialLR checkpoint payload: scheduler count mismatch (expected "
                                + schedulers.size() + ", got " + childStates.size() + ")");
            }
            for (int i = 0; i < childStates.size(); i++) {
                Object childState = childStates.get(i);
                if (!(childState instanceof Map<?, ?> rawChildMap)) {
                    throw new IllegalArgumentException(
                            "Invalid SequentialLR checkpoint payload: scheduler state " + i + " must be a Map");
                }
                schedulers.get(i).loadStateDict((Map<String, Object>) rawChildMap);
            }
        }
        currentStep = SchedulerValidation.readNonNegativeInt(
                state.get("currentStep"), currentStep, "SequentialLR", "currentStep");
        activeIndex = SchedulerValidation.readNonNegativeInt(
                state.get("activeIndex"), schedulerIndexForStep(currentStep), "SequentialLR", "activeIndex");
        if (activeIndex >= schedulers.size()) {
            throw new IllegalArgumentException(
                    "Invalid SequentialLR checkpoint payload: activeIndex out of range, got " + activeIndex);
        }
        int expectedActive = schedulerIndexForStep(currentStep);
        if (activeIndex != expectedActive) {
            throw new IllegalArgumentException(
                    "Invalid SequentialLR checkpoint payload: activeIndex mismatch (expected "
                            + expectedActive + ", got " + activeIndex + ")");
        }
        currentLr = SchedulerValidation.readLearningRate(
                state.get("currentLr"), schedulers.get(activeIndex).getLr(), "SequentialLR", "currentLr");
        setLearningRate(currentLr);
    }

    private int schedulerIndexForStep(int completedSteps) {
        int index = 0;
        while (index < milestones.length && completedSteps >= milestones[index]) {
            index++;
        }
        return index;
    }

    private void requireMilestonesMatch(Object rawMilestones) {
        if (rawMilestones == null) {
            return;
        }
        int[] loaded = coerceMilestones(rawMilestones);
        if (!Arrays.equals(loaded, milestones)) {
            throw new IllegalArgumentException(
                    "Invalid SequentialLR checkpoint payload: milestones mismatch (expected "
                            + Arrays.toString(milestones) + ", got " + Arrays.toString(loaded) + ")");
        }
    }

    private static List<LRScheduler> validateSchedulers(Optimizer optimizer, List<LRScheduler> schedulers) {
        List<LRScheduler> validated = new ArrayList<>(schedulers.size());
        for (int i = 0; i < schedulers.size(); i++) {
            LRScheduler scheduler = schedulers.get(i);
            if (scheduler == null) {
                throw new IllegalArgumentException("scheduler " + i + " must not be null");
            }
            if (scheduler.optimizer != optimizer) {
                throw new IllegalArgumentException("scheduler " + i + " must use the same optimizer");
            }
            validated.add(scheduler);
        }
        return validated;
    }

    private static int[] validateMilestones(int[] milestones) {
        int[] copy = Arrays.copyOf(milestones, milestones.length);
        int previous = 0;
        for (int milestone : copy) {
            if (milestone <= previous) {
                throw new IllegalArgumentException(
                        "milestones must be strictly increasing positive steps, got: "
                                + Arrays.toString(milestones));
            }
            previous = milestone;
        }
        return copy;
    }

    private static int[] coerceMilestones(Object rawMilestones) {
        if (rawMilestones instanceof int[] values) {
            return Arrays.copyOf(values, values.length);
        }
        if (rawMilestones instanceof List<?> list) {
            int[] values = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object value = list.get(i);
                if (!(value instanceof Number number)) {
                    throw new IllegalArgumentException(
                            "Invalid SequentialLR checkpoint payload: milestones must contain numbers");
                }
                values[i] = number.intValue();
            }
            return values;
        }
        throw new IllegalArgumentException(
                "Invalid SequentialLR checkpoint payload: milestones must be an int[] or List");
    }
}
