package tech.kayys.gollek.ml.optim;

import java.util.HashMap;
import java.util.Map;

/**
 * Reduces the optimizer learning rate when a validation loss or metric stops
 * improving.
 *
 * <p>This scheduler is meant to be stepped after validation with
 * {@link #step(double)}. The first finite metric initializes the best value.
 * Later non-improving values increment the bad-step counter; once that counter
 * exceeds {@code patience}, the current learning rate is multiplied by
 * {@code factor} and clamped to {@code minLr}.</p>
 */
public final class ReduceLROnPlateau extends LRScheduler {

    public enum Mode {
        MIN {
            @Override
            boolean isImproved(double current, double best, double threshold) {
                return current < best - threshold;
            }
        },
        MAX {
            @Override
            boolean isImproved(double current, double best, double threshold) {
                return current > best + threshold;
            }
        };

        abstract boolean isImproved(double current, double best, double threshold);
    }

    private final Mode mode;
    private final float factor;
    private final int patience;
    private final double threshold;
    private final int cooldown;
    private final float minLr;

    private int stepCount;
    private int badSteps;
    private int cooldownRemaining;
    private int reductionCount;
    private double bestMetric = Double.NaN;
    private float currentLr;

    public ReduceLROnPlateau(
            Optimizer optimizer,
            Mode mode,
            float factor,
            int patience,
            double threshold,
            int cooldown,
            float minLr) {
        super(optimizer);
        if (factor <= 0.0f || factor >= 1.0f) {
            throw new IllegalArgumentException("factor must be in (0, 1), got: " + factor);
        }
        if (patience < 0) {
            throw new IllegalArgumentException("patience must be non-negative, got: " + patience);
        }
        if (!Double.isFinite(threshold) || threshold < 0.0) {
            throw new IllegalArgumentException("threshold must be finite and non-negative, got: " + threshold);
        }
        if (cooldown < 0) {
            throw new IllegalArgumentException("cooldown must be non-negative, got: " + cooldown);
        }
        if (minLr < 0.0f) {
            throw new IllegalArgumentException("minLr must be non-negative, got: " + minLr);
        }
        this.mode = mode == null ? Mode.MIN : mode;
        this.factor = factor;
        this.patience = patience;
        this.threshold = threshold;
        this.cooldown = cooldown;
        this.minLr = minLr;
        this.currentLr = optimizer.learningRate();
    }

    @Override
    public void step() {
        step(Double.NaN);
    }

    @Override
    public void step(double metric) {
        stepCount++;
        if (!Double.isFinite(metric)) {
            return;
        }

        if (Double.isNaN(bestMetric) || mode.isImproved(metric, bestMetric, threshold)) {
            bestMetric = metric;
            badSteps = 0;
            return;
        }

        if (cooldownRemaining > 0) {
            cooldownRemaining--;
            badSteps = 0;
            return;
        }

        badSteps++;
        if (badSteps > patience) {
            reduceLearningRate();
            cooldownRemaining = cooldown;
            badSteps = 0;
        }
    }

    @Override
    public float getLr() {
        return currentLr;
    }

    public int stepCount() {
        return stepCount;
    }

    public int badSteps() {
        return badSteps;
    }

    public int reductionCount() {
        return reductionCount;
    }

    public double bestMetric() {
        return bestMetric;
    }

    private void reduceLearningRate() {
        float nextLr = Math.max(minLr, currentLr * factor);
        if (nextLr < currentLr - 1.0e-12f) {
            currentLr = nextLr;
            setLearningRate(currentLr);
            reductionCount++;
        } else {
            currentLr = optimizer.learningRate();
        }
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("scheduler", "ReduceLROnPlateau");
        state.put("mode", mode.name());
        state.put("factor", factor);
        state.put("patience", patience);
        state.put("threshold", threshold);
        state.put("cooldown", cooldown);
        state.put("minLr", minLr);
        state.put("stepCount", stepCount);
        state.put("badSteps", badSteps);
        state.put("cooldownRemaining", cooldownRemaining);
        state.put("reductionCount", reductionCount);
        state.put("bestMetric", bestMetric);
        state.put("currentLr", currentLr);
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }
        Object schedulerName = state.get("scheduler");
        if (schedulerName instanceof String name && !"ReduceLROnPlateau".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint scheduler mismatch: expected ReduceLROnPlateau but got " + name);
        }
        requireModeMatch(state.get("mode"));
        requireFloatMatch(state.get("factor"), factor, "factor");
        requireIntMatch(state.get("patience"), patience, "patience");
        requireDoubleMatch(state.get("threshold"), threshold, "threshold");
        requireIntMatch(state.get("cooldown"), cooldown, "cooldown");
        requireFloatMatch(state.get("minLr"), minLr, "minLr");
        stepCount = Math.max(0, readInt(state.get("stepCount"), stepCount));
        badSteps = Math.max(0, readInt(state.get("badSteps"), badSteps));
        cooldownRemaining = Math.max(0, readInt(state.get("cooldownRemaining"), cooldownRemaining));
        reductionCount = Math.max(0, readInt(state.get("reductionCount"), reductionCount));
        bestMetric = readDouble(state.get("bestMetric"), bestMetric);
        currentLr = readFloat(state.get("currentLr"), currentLr);
        setLearningRate(currentLr);
    }

    private void requireModeMatch(Object value) {
        if (value == null) {
            return;
        }
        Mode loaded = Mode.valueOf(String.valueOf(value));
        if (loaded != mode) {
            throw new IllegalArgumentException(
                    "Invalid ReduceLROnPlateau checkpoint payload: mode mismatch (expected "
                            + mode + ", got " + loaded + ")");
        }
    }

    private static void requireIntMatch(Object value, int expected, String fieldName) {
        if (value == null) {
            return;
        }
        int loaded = readInt(value, expected);
        if (loaded != expected) {
            throw new IllegalArgumentException(
                    "Invalid ReduceLROnPlateau checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static void requireFloatMatch(Object value, float expected, String fieldName) {
        if (value == null) {
            return;
        }
        float loaded = readFloat(value, expected);
        if (Math.abs(loaded - expected) > 1e-7f) {
            throw new IllegalArgumentException(
                    "Invalid ReduceLROnPlateau checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static void requireDoubleMatch(Object value, double expected, String fieldName) {
        if (value == null) {
            return;
        }
        double loaded = readDouble(value, expected);
        if (Math.abs(loaded - expected) > 1e-12) {
            throw new IllegalArgumentException(
                    "Invalid ReduceLROnPlateau checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static int readInt(Object value, int fallback) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static float readFloat(Object value, float fallback) {
        if (value instanceof Number number) {
            return number.floatValue();
        }
        if (value instanceof String text) {
            try {
                return Float.parseFloat(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }

    private static double readDouble(Object value, double fallback) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text) {
            try {
                return Double.parseDouble(text);
            } catch (NumberFormatException ignored) {
                return fallback;
            }
        }
        return fallback;
    }
}
