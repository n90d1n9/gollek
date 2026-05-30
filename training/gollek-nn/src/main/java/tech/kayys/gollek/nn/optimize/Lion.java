package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Lion optimizer — discovered by Google Brain via program search.
 *
 * <p>
 * Based on <em>"Symbolic Discovery of Optimization Algorithms"</em>
 * (Chen et al., 2023). Uses only the sign of the gradient update,
 * making it memory-efficient (no second moment) and often faster than Adam.
 *
 * <p>
 * Update rule:
 * 
 * <pre>
 *   c_t = β₁·m_{t-1} + (1-β₁)·g
 *   θ  -= lr · (sign(c_t) + λ·θ)
 *   m_t = β₂·m_{t-1} + (1-β₂)·g
 * </pre>
 *
 * <p>
 * Uses JDK 25 Vector API via {@link VectorOps} for the sign + update loop.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var optimizer = new Lion(model.parameters(), lr = 1e-4f, weightDecay = 0.01f);
 * }</pre>
 */
public final class Lion implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float beta1;
    private final float beta2;
    private final float weightDecay;
    private final Map<Parameter, float[]> momentum = new HashMap<>();
    private int step;

    /**
     * Creates a Lion optimizer with default hyperparameters.
     *
     * @param parameters model parameters
     * @param lr         learning rate (typically 3-10× smaller than Adam)
     */
    public Lion(List<Parameter> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.99f, 0.0f);
    }

    /**
     * Creates a Lion optimizer with full control.
     *
     * @param parameters  model parameters
     * @param lr          learning rate
     * @param beta1       interpolation for update direction (default 0.9)
     * @param beta2       momentum decay (default 0.99)
     * @param weightDecay L2 regularization coefficient
     */
    public Lion(List<Parameter> parameters, float lr,
            float beta1, float beta2, float weightDecay) {
        this.parameters = OptimizerValidation.requireParameters(parameters);
        this.lr = OptimizerValidation.learningRate(lr);
        this.beta1 = OptimizerValidation.beta(beta1, "beta1");
        this.beta2 = OptimizerValidation.beta(beta2, "beta2");
        this.weightDecay = OptimizerValidation.weightDecay(weightDecay);
    }

    @Override
    public void step() {
        OptimizerValidation.requireStepInputs(parameters, "Lion");
        step++;
        for (Parameter p : parameters) {
            if (p.data().grad() == null)
                continue;
            float[] theta = p.data().data();
            float[] grad = p.data().grad().data();
            int len = theta.length;
            float[] m = momentum.computeIfAbsent(p, k -> new float[len]);

            for (int i = 0; i < len; i++) {
                // c = β₁·m + (1-β₁)·g
                float c = beta1 * m[i] + (1f - beta1) * grad[i];
                // θ -= lr · (sign(c) + λ·θ)
                theta[i] -= lr * (Math.signum(c) + weightDecay * theta[i]);
                // m = β₂·m + (1-β₂)·g
                m[i] = beta2 * m[i] + (1f - beta2) * grad[i];
            }
        }
    }

    @Override
    public void zeroGrad() {
        parameters.forEach(p -> p.data().zeroGrad());
    }

    @Override
    public float learningRate() {
        return lr;
    }

    @Override
    public List<Parameter> parameters() {
        return parameters;
    }

    @Override
    public void setLearningRate(float lr) {
        this.lr = OptimizerValidation.learningRate(lr);
    }

    public int stepCount() {
        return step;
    }

    public float beta1() {
        return beta1;
    }

    public float beta2() {
        return beta2;
    }

    public float weightDecay() {
        return weightDecay;
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("optimizer", "Lion");
        state.put("step", step);
        state.put("learningRate", lr);
        state.put("beta1", beta1);
        state.put("beta2", beta2);
        state.put("weightDecay", weightDecay);
        state.put("momentum", exportMomentum());
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }

        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"Lion".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected Lion but got " + name);
        }

        requireFloatMatch(state.get("beta1"), beta1, "beta1");
        requireFloatMatch(state.get("beta2"), beta2, "beta2");
        requireFloatMatch(state.get("weightDecay"), weightDecay, "weightDecay");
        step = Math.max(0, readInt(state.get("step"), step));
        lr = OptimizerValidation.learningRate(readFloat(state.get("learningRate"), lr));

        Object momentumState = state.get("momentum");
        if (momentumState == null) {
            throw new IllegalArgumentException("Invalid Lion checkpoint payload: missing key 'momentum'");
        }
        restoreMomentum(momentumState);
    }

    private List<float[]> exportMomentum() {
        List<float[]> exported = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) {
            float[] source = momentum.get(parameter);
            exported.add(source == null ? new float[(int) parameter.data().numel()] : source.clone());
        }
        return exported;
    }

    private void restoreMomentum(Object rawMomentumState) {
        if (!(rawMomentumState instanceof List<?> serializedMomentum)) {
            throw new IllegalArgumentException("Invalid Lion checkpoint payload: momentum must be a List");
        }
        if (serializedMomentum.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid Lion checkpoint payload: momentum size mismatch (expected "
                            + parameters.size() + ", got " + serializedMomentum.size() + ")");
        }

        for (int i = 0; i < parameters.size(); i++) {
            float[] source = coerceFloatArray(serializedMomentum.get(i), i);
            Parameter parameter = parameters.get(i);
            float[] target = momentum.get(parameter);
            if (target == null) {
                target = new float[(int) parameter.data().numel()];
                momentum.put(parameter, target);
            }
            if (source.length != target.length) {
                throw new IllegalArgumentException(
                        "Invalid Lion checkpoint payload: momentum[" + i
                                + "] length mismatch (expected " + target.length
                                + ", got " + source.length + ")");
            }
            System.arraycopy(source, 0, target, 0, source.length);
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

    private static void requireFloatMatch(Object value, float expected, String fieldName) {
        if (value == null) {
            return;
        }
        float loaded = readFloat(value, expected);
        if (!Float.isFinite(loaded) || Math.abs(loaded - expected) > 1e-7f) {
            throw new IllegalArgumentException(
                    "Invalid Lion checkpoint payload: " + fieldName
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
                            "Invalid Lion checkpoint payload: momentum[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid Lion checkpoint payload: momentum[" + index + "] must be float[]");
    }
}
