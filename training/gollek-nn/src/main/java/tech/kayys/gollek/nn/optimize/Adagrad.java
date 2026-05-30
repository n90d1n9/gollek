package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adagrad optimizer — adapts learning rates per parameter based on the sum
 * of squared historical gradients.
 *
 * <p>
 * Parameters that receive large gradients get smaller effective learning rates,
 * making Adagrad well-suited for sparse features (NLP, recommendation systems).
 *
 * <p>
 * Update rule:
 * 
 * <pre>
 *   G_t = G_{t-1} + g²
 *   θ  -= lr / (√G_t + ε) · g
 * </pre>
 *
 * <p>
 * Uses JDK 25 Vector API via {@link VectorOps} for the update loop.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var optimizer = new Adagrad(model.parameters(), lr = 0.01f);
 * }</pre>
 */
public final class Adagrad implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float eps;
    private final float weightDecay;
    private final Map<Parameter, float[]> sumSq = new HashMap<>();
    private int step;

    /**
     * Creates an Adagrad optimizer with default epsilon.
     *
     * @param parameters model parameters
     * @param lr         learning rate (default 0.01 for Adagrad)
     */
    public Adagrad(List<Parameter> parameters, float lr) {
        this(parameters, lr, 1e-8f, 0f);
    }

    /**
     * Creates an Adagrad optimizer with full control.
     *
     * @param parameters  model parameters
     * @param lr          learning rate
     * @param eps         numerical stability constant (default 1e-8)
     * @param weightDecay L2 regularization coefficient
     */
    public Adagrad(List<Parameter> parameters, float lr, float eps, float weightDecay) {
        this.parameters = OptimizerValidation.requireParameters(parameters);
        this.lr = OptimizerValidation.learningRate(lr);
        this.eps = OptimizerValidation.epsilon(eps);
        this.weightDecay = OptimizerValidation.weightDecay(weightDecay);
    }

    @Override
    public void step() {
        OptimizerValidation.requireStepInputs(parameters, "Adagrad");
        step++;
        for (Parameter p : parameters) {
            if (p.data().grad() == null)
                continue;
            float[] theta = p.data().data();
            float[] grad = p.data().grad().data();
            int len = theta.length;
            float[] G = sumSq.computeIfAbsent(p, k -> new float[len]);

            // Keep weight decay consistent for every element instead of only the scalar tail.
            for (int i = 0; i < len; i++) {
                float g = grad[i] + weightDecay * theta[i];
                G[i] += g * g;
                theta[i] -= lr / ((float) Math.sqrt(G[i]) + eps) * g;
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

    public float epsilon() {
        return eps;
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
        state.put("optimizer", "Adagrad");
        state.put("step", step);
        state.put("learningRate", lr);
        state.put("epsilon", eps);
        state.put("weightDecay", weightDecay);
        state.put("sumSq", exportSlot(sumSq));
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }

        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"Adagrad".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected Adagrad but got " + name);
        }

        requireFloatMatch(state.get("epsilon"), eps, "epsilon");
        requireFloatMatch(state.get("weightDecay"), weightDecay, "weightDecay");
        step = Math.max(0, readInt(state.get("step"), step));
        lr = OptimizerValidation.learningRate(readFloat(state.get("learningRate"), lr));

        Object sumSqState = state.get("sumSq");
        if (sumSqState == null) {
            throw new IllegalArgumentException("Invalid Adagrad checkpoint payload: missing key 'sumSq'");
        }
        restoreSlot(sumSq, sumSqState, "sumSq");
    }

    private List<float[]> exportSlot(Map<Parameter, float[]> slotMap) {
        List<float[]> exported = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) {
            float[] source = slotMap.get(parameter);
            exported.add(source == null ? new float[(int) parameter.data().numel()] : source.clone());
        }
        return exported;
    }

    private void restoreSlot(Map<Parameter, float[]> slotMap, Object rawState, String slotName) {
        if (!(rawState instanceof List<?> serializedSlot)) {
            throw new IllegalArgumentException(
                    "Invalid Adagrad checkpoint payload: " + slotName + " must be a List");
        }
        if (serializedSlot.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid Adagrad checkpoint payload: " + slotName + " size mismatch (expected "
                            + parameters.size() + ", got " + serializedSlot.size() + ")");
        }

        for (int i = 0; i < parameters.size(); i++) {
            float[] source = coerceFloatArray(serializedSlot.get(i), slotName, i);
            Parameter parameter = parameters.get(i);
            float[] target = slotMap.get(parameter);
            if (target == null) {
                target = new float[(int) parameter.data().numel()];
                slotMap.put(parameter, target);
            }
            if (source.length != target.length) {
                throw new IllegalArgumentException(
                        "Invalid Adagrad checkpoint payload: " + slotName + "[" + i
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
                    "Invalid Adagrad checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static float[] coerceFloatArray(Object value, String slotName, int index) {
        if (value instanceof float[] array) {
            return array;
        }
        if (value instanceof List<?> numbers) {
            float[] converted = new float[numbers.size()];
            for (int i = 0; i < numbers.size(); i++) {
                Object element = numbers.get(i);
                if (!(element instanceof Number number)) {
                    throw new IllegalArgumentException(
                            "Invalid Adagrad checkpoint payload: " + slotName + "[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid Adagrad checkpoint payload: " + slotName + "[" + index + "] must be float[]");
    }
}
