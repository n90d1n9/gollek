package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.nn.Parameter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Lookahead optimizer — wraps any base optimizer with a slow-weights update.
 *
 * <p>Based on <em>"Lookahead Optimizer: k steps forward, 1 step back"</em>
 * (Zhang et al., 2019). Improves convergence stability by maintaining slow
 * weights that interpolate toward fast weights every k steps.
 *
 * <p>Update rule (every k steps):
 * <pre>
 *   slow_weights += alpha * (fast_weights - slow_weights)
 *   fast_weights  = slow_weights
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var base = Adam.builder(model.parameters(), 0.001f).build();
 * var optimizer = new Lookahead(base, k=5, alpha=0.5f);
 * }</pre>
 */
public final class Lookahead implements Optimizer {

    private final Optimizer inner;
    private final List<Parameter> parameters;
    private final int k;
    private final float alpha;
    private int stepCount = 0;

    /** Slow weights snapshot per parameter (flat index → slow weight array). */
    private final Map<Parameter, float[]> slowWeights = new HashMap<>();

    /**
     * Wraps a base optimizer with Lookahead.
     *
     * @param inner base optimizer (Adam, SGD, etc.)
     * @param k     number of inner steps before slow-weight update (default 5)
     * @param alpha slow-weight interpolation factor (default 0.5)
     */
    public Lookahead(Optimizer inner, int k, float alpha) {
        if (inner == null) {
            throw new IllegalArgumentException("inner optimizer must not be null");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got: " + k);
        }
        if (!Float.isFinite(alpha) || alpha <= 0.0f || alpha > 1.0f) {
            throw new IllegalArgumentException("alpha must be finite and in (0, 1], got: " + alpha);
        }
        this.inner = inner;
        this.parameters = List.copyOf(OptimizerValidation.requireParameters(inner.parameters()));
        this.k = k;
        this.alpha = alpha;
        initializeSlowWeights();
    }

    /** Wraps with default k=5, alpha=0.5. */
    public Lookahead(Optimizer inner) { this(inner, 5, 0.5f); }

    @Override
    public void step() {
        inner.step();
        stepCount++;

        if (stepCount % k == 0) {
            synchronizeSlowWeights();
        }
    }

    @Override public void zeroGrad()                { inner.zeroGrad(); }
    @Override public float learningRate()           { return inner.learningRate(); }
    @Override public void setLearningRate(float lr) { inner.setLearningRate(lr); }

    /** @return the wrapped base optimizer */
    public Optimizer inner() { return inner; }

    /** Exposes parameters from the inner optimizer. */
    public List<Parameter> parameters() { return parameters; }

    public int k() {
        return k;
    }

    public float alpha() {
        return alpha;
    }

    public int stepCount() {
        return stepCount;
    }

    @Override
    public boolean supportsStateDict() {
        return inner.supportsStateDict();
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("optimizer", "Lookahead");
        state.put("innerOptimizerClass", inner.getClass().getName());
        state.put("k", k);
        state.put("alpha", alpha);
        state.put("stepCount", stepCount);
        state.put("slowWeights", exportSlowWeights());
        if (inner.supportsStateDict()) {
            state.put("innerState", inner.stateDict());
        }
        return state;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }
        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"Lookahead".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected Lookahead but got " + name);
        }
        Object innerClass = state.get("innerOptimizerClass");
        if (innerClass instanceof String className && !inner.getClass().getName().equals(className)) {
            throw new IllegalArgumentException(
                    "Invalid Lookahead checkpoint payload: inner optimizer mismatch (expected "
                            + inner.getClass().getName() + ", got " + className + ")");
        }
        requireIntMatch(state.get("k"), k, "k");
        requireFloatMatch(state.get("alpha"), alpha, "alpha");
        stepCount = readNonNegativeInt(state.get("stepCount"), stepCount, "stepCount");

        Object slowState = state.get("slowWeights");
        if (slowState == null) {
            throw new IllegalArgumentException("Invalid Lookahead checkpoint payload: missing key 'slowWeights'");
        }
        restoreSlowWeights(slowState);

        if (inner.supportsStateDict()) {
            Object innerState = state.get("innerState");
            if (innerState == null) {
                throw new IllegalArgumentException("Invalid Lookahead checkpoint payload: missing key 'innerState'");
            }
            if (!(innerState instanceof Map<?, ?> rawMap)) {
                throw new IllegalArgumentException("Invalid Lookahead checkpoint payload: innerState must be a Map");
            }
            inner.loadStateDict((Map<String, Object>) rawMap);
        }
    }

    private void initializeSlowWeights() {
        slowWeights.clear();
        for (Parameter parameter : parameters) {
            slowWeights.put(parameter, parameter.data().data().clone());
        }
    }

    private void synchronizeSlowWeights() {
        for (Parameter parameter : parameters) {
            float[] fast = parameter.data().data();
            float[] slow = slowWeights.get(parameter);
            if (slow == null || slow.length != fast.length) {
                slow = fast.clone();
                slowWeights.put(parameter, slow);
            }
            for (int i = 0; i < fast.length; i++) {
                slow[i] += alpha * (fast[i] - slow[i]);
                fast[i] = slow[i];
            }
        }
    }

    private List<float[]> exportSlowWeights() {
        List<float[]> exported = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) {
            float[] source = slowWeights.get(parameter);
            exported.add(source == null ? parameter.data().data().clone() : source.clone());
        }
        return exported;
    }

    private void restoreSlowWeights(Object rawSlowWeights) {
        if (!(rawSlowWeights instanceof List<?> serializedSlowWeights)) {
            throw new IllegalArgumentException("Invalid Lookahead checkpoint payload: slowWeights must be a List");
        }
        if (serializedSlowWeights.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid Lookahead checkpoint payload: slowWeights size mismatch (expected "
                            + parameters.size() + ", got " + serializedSlowWeights.size() + ")");
        }

        for (int i = 0; i < parameters.size(); i++) {
            float[] source = coerceFloatArray(serializedSlowWeights.get(i), i);
            Parameter parameter = parameters.get(i);
            float[] fast = parameter.data().data();
            if (source.length != fast.length) {
                throw new IllegalArgumentException(
                        "Invalid Lookahead checkpoint payload: slowWeights[" + i
                                + "] length mismatch (expected " + fast.length
                                + ", got " + source.length + ")");
            }
            slowWeights.put(parameter, source.clone());
        }
    }

    private static int readNonNegativeInt(Object value, int fallback, String fieldName) {
        int loaded = readInt(value, fallback);
        if (loaded < 0) {
            throw new IllegalArgumentException(
                    "Invalid Lookahead checkpoint payload: " + fieldName + " must be non-negative");
        }
        return loaded;
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

    private static void requireIntMatch(Object value, int expected, String fieldName) {
        if (value == null) {
            return;
        }
        int loaded = readInt(value, expected);
        if (loaded != expected) {
            throw new IllegalArgumentException(
                    "Invalid Lookahead checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static void requireFloatMatch(Object value, float expected, String fieldName) {
        if (value == null) {
            return;
        }
        float loaded = readFloat(value, expected);
        if (!Float.isFinite(loaded) || Math.abs(loaded - expected) > 1e-7f) {
            throw new IllegalArgumentException(
                    "Invalid Lookahead checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static float[] coerceFloatArray(Object value, int index) {
        if (value instanceof float[] array) {
            return array;
        }
        if (value instanceof List<?> numbers) {
            float[] converted = new float[numbers.size()];
            for (int i = 0; i < numbers.size(); i++) {
                Object element = numbers.get(i);
                if (!(element instanceof Number number)) {
                    throw new IllegalArgumentException(
                            "Invalid Lookahead checkpoint payload: slowWeights[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid Lookahead checkpoint payload: slowWeights[" + index + "] must be float[]");
    }
}
