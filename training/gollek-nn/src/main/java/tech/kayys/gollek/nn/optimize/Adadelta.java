package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adadelta optimizer — adapts learning rates without requiring a global lr.
 *
 * <p>
 * Based on <em>"ADADELTA: An Adaptive Learning Rate Method"</em> (Zeiler,
 * 2012).
 * Addresses Adagrad's monotonically decreasing learning rate by using a
 * window of accumulated gradients instead of all past gradients.
 *
 * <p>
 * Update rule:
 * 
 * <pre>
 *   E[g²]_t = ρ·E[g²]_{t-1} + (1-ρ)·g²
 *   Δθ = -√(E[Δθ²]_{t-1} + ε) / √(E[g²]_t + ε) · g
 *   E[Δθ²]_t = ρ·E[Δθ²]_{t-1} + (1-ρ)·Δθ²
 *   θ += Δθ
 * </pre>
 *
 * <p>
 * Uses JDK 25 Vector API via {@link VectorOps} for the update loop.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var optimizer = new Adadelta(model.parameters()); // no lr needed
 * }</pre>
 */
public final class Adadelta implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float rho;
    private final float eps;
    private final Map<Parameter, float[]> eGrad = new HashMap<>(); // E[g²]
    private final Map<Parameter, float[]> eDelta = new HashMap<>(); // E[Δθ²]
    private int step;

    /**
     * Creates an Adadelta optimizer with default hyperparameters.
     * No learning rate tuning required.
     *
     * @param parameters model parameters
     */
    public Adadelta(List<Parameter> parameters) {
        this(parameters, 1.0f, 0.95f, 1e-6f);
    }

    /**
     * Creates an Adadelta optimizer with full control.
     *
     * @param parameters model parameters
     * @param lr         scaling factor (default 1.0 — Adadelta is lr-free)
     * @param rho        decay rate for running averages (default 0.95)
     * @param eps        numerical stability constant (default 1e-6)
     */
    public Adadelta(List<Parameter> parameters, float lr, float rho, float eps) {
        this.parameters = OptimizerValidation.requireParameters(parameters);
        this.lr = OptimizerValidation.learningRate(lr);
        this.rho = OptimizerValidation.beta(rho, "rho");
        this.eps = OptimizerValidation.epsilon(eps);
    }

    @Override
    public void step() {
        OptimizerValidation.requireStepInputs(parameters, "Adadelta");
        step++;
        for (Parameter p : parameters) {
            if (p.data().grad() == null)
                continue;
            float[] theta = p.data().data();
            float[] grad = p.data().grad().data();
            int len = theta.length;
            float[] eg = eGrad.computeIfAbsent(p, k -> new float[len]);
            float[] ed = eDelta.computeIfAbsent(p, k -> new float[len]);

            for (int i = 0; i < len; i++) {
                eg[i] = rho * eg[i] + (1f - rho) * grad[i] * grad[i];
                float rmsG = (float) Math.sqrt(eg[i] + eps);
                float rmsD = (float) Math.sqrt(ed[i] + eps);
                float delta = -(rmsD / rmsG) * grad[i];
                ed[i] = rho * ed[i] + (1f - rho) * delta * delta;
                theta[i] += lr * delta;
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

    public float rho() {
        return rho;
    }

    public float epsilon() {
        return eps;
    }

    @Override
    public boolean supportsStateDict() {
        return true;
    }

    @Override
    public Map<String, Object> stateDict() {
        Map<String, Object> state = new HashMap<>();
        state.put("optimizer", "Adadelta");
        state.put("step", step);
        state.put("learningRate", lr);
        state.put("rho", rho);
        state.put("epsilon", eps);
        state.put("eGrad", exportSlot(eGrad));
        state.put("eDelta", exportSlot(eDelta));
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }

        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"Adadelta".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected Adadelta but got " + name);
        }

        requireFloatMatch(state.get("rho"), rho, "rho");
        requireFloatMatch(state.get("epsilon"), eps, "epsilon");
        step = Math.max(0, readInt(state.get("step"), step));
        lr = OptimizerValidation.learningRate(readFloat(state.get("learningRate"), lr));

        Object eGradState = state.get("eGrad");
        if (eGradState == null) {
            throw new IllegalArgumentException("Invalid Adadelta checkpoint payload: missing key 'eGrad'");
        }
        restoreSlot(eGrad, eGradState, "eGrad");

        Object eDeltaState = state.get("eDelta");
        if (eDeltaState == null) {
            throw new IllegalArgumentException("Invalid Adadelta checkpoint payload: missing key 'eDelta'");
        }
        restoreSlot(eDelta, eDeltaState, "eDelta");
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
                    "Invalid Adadelta checkpoint payload: " + slotName + " must be a List");
        }
        if (serializedSlot.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid Adadelta checkpoint payload: " + slotName + " size mismatch (expected "
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
                        "Invalid Adadelta checkpoint payload: " + slotName + "[" + i
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
                    "Invalid Adadelta checkpoint payload: " + fieldName
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
                            "Invalid Adadelta checkpoint payload: " + slotName + "[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid Adadelta checkpoint payload: " + slotName + "[" + index + "] must be float[]");
    }
}
