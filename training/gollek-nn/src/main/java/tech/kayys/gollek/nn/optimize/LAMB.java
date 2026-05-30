package tech.kayys.gollek.ml.optim;

import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.autograd.VectorOps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * LAMB optimizer — Layer-wise Adaptive Moments for Batch training.
 *
 * <p>
 * Designed for large-batch training (e.g. BERT pre-training with batch size
 * 32K+).
 * Extends Adam with a layer-wise trust ratio that scales the update per layer.
 *
 * <p>
 * Based on <em>"Large Batch Optimization for Deep Learning: Training BERT in 76
 * minutes"</em>
 * (You et al., 2019).
 *
 * <p>
 * Update rule:
 * 
 * <pre>
 *   m_t = β₁·m_{t-1} + (1-β₁)·g
 *   v_t = β₂·v_{t-1} + (1-β₂)·g²
 *   m̂ = m_t / (1-β₁ᵗ),  v̂ = v_t / (1-β₂ᵗ)
 *   r = m̂ / (√v̂ + ε) + λ·θ          (Adam update + weight decay)
 *   trust = ||θ|| / ||r||              (layer-wise trust ratio)
 *   θ -= lr · trust · r
 * </pre>
 *
 * <p>
 * Uses JDK 25 Vector API via {@link VectorOps} for the inner update loops.
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * var optimizer = new LAMB(model.parameters(), lr = 1e-3f, weightDecay = 0.01f);
 * }</pre>
 */
public final class LAMB implements Optimizer {

    private final List<Parameter> parameters;
    private float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;

    private final Map<Parameter, float[]> m = new HashMap<>(); // 1st moment
    private final Map<Parameter, float[]> v = new HashMap<>(); // 2nd moment
    private int step = 0;

    /**
     * Creates a LAMB optimizer with default hyperparameters.
     *
     * @param parameters model parameters to optimize
     * @param lr         learning rate
     */
    public LAMB(List<Parameter> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.999f, 1e-6f, 0.01f);
    }

    /**
     * Creates a LAMB optimizer with full hyperparameter control.
     *
     * @param parameters  model parameters
     * @param lr          learning rate
     * @param beta1       first moment decay (default 0.9)
     * @param beta2       second moment decay (default 0.999)
     * @param eps         numerical stability constant (default 1e-6)
     * @param weightDecay L2 regularization coefficient (default 0.01)
     */
    public LAMB(List<Parameter> parameters, float lr,
            float beta1, float beta2, float eps, float weightDecay) {
        this.parameters = OptimizerValidation.requireParameters(parameters);
        this.lr = OptimizerValidation.learningRate(lr);
        this.beta1 = OptimizerValidation.beta(beta1, "beta1");
        this.beta2 = OptimizerValidation.beta(beta2, "beta2");
        this.eps = OptimizerValidation.epsilon(eps);
        this.weightDecay = OptimizerValidation.weightDecay(weightDecay);
    }

    @Override
    public void step() {
        OptimizerValidation.requireStepInputs(parameters, "LAMB");
        step++;
        float bc1 = 1f - (float) Math.pow(beta1, step);
        float bc2 = 1f - (float) Math.pow(beta2, step);

        for (Parameter p : parameters) {
            if (p.data().grad() == null)
                continue;
            float[] theta = p.data().data();
            float[] grad = p.data().grad().data();
            int len = theta.length;

            float[] mt = m.computeIfAbsent(p, k -> new float[len]);
            float[] vt = v.computeIfAbsent(p, k -> new float[len]);

            // Adam moments + bias correction
            float[] r = new float[len];
            for (int i = 0; i < len; i++) {
                mt[i] = beta1 * mt[i] + (1f - beta1) * grad[i];
                vt[i] = beta2 * vt[i] + (1f - beta2) * grad[i] * grad[i];
                float mHat = mt[i] / bc1;
                float vHat = vt[i] / bc2;
                r[i] = mHat / ((float) Math.sqrt(vHat) + eps) + weightDecay * theta[i];
            }

            // Layer-wise trust ratio: ||θ|| / ||r||
            float[] thetaSq = new float[len], rSq = new float[len];
            VectorOps.mul(theta, theta, thetaSq);
            VectorOps.mul(r, r, rSq);
            float normTheta = (float) Math.sqrt(VectorOps.sum(thetaSq));
            float normR = (float) Math.sqrt(VectorOps.sum(rSq));
            float trust = (normTheta > 0 && normR > 0) ? normTheta / normR : 1f;

            // Update: θ -= lr * trust * r (SIMD via VectorOps)
            VectorOps.mulScalar(r, lr * trust, r);
            float[] neg = new float[len];
            VectorOps.mulScalar(r, -1f, neg);
            VectorOps.add(theta, neg, theta);
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
        state.put("optimizer", "LAMB");
        state.put("step", step);
        state.put("learningRate", lr);
        state.put("beta1", beta1);
        state.put("beta2", beta2);
        state.put("epsilon", eps);
        state.put("weightDecay", weightDecay);
        state.put("m", exportSlot(m));
        state.put("v", exportSlot(v));
        return state;
    }

    @Override
    public void loadStateDict(Map<String, Object> state) {
        if (state == null || state.isEmpty()) {
            return;
        }

        Object optimizerName = state.get("optimizer");
        if (optimizerName instanceof String name && !"LAMB".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected LAMB but got " + name);
        }

        requireFloatMatch(state.get("beta1"), beta1, "beta1");
        requireFloatMatch(state.get("beta2"), beta2, "beta2");
        requireFloatMatch(state.get("epsilon"), eps, "epsilon");
        requireFloatMatch(state.get("weightDecay"), weightDecay, "weightDecay");
        step = Math.max(0, readInt(state.get("step"), step));
        lr = OptimizerValidation.learningRate(readFloat(state.get("learningRate"), lr));

        Object mState = state.get("m");
        if (mState == null) {
            throw new IllegalArgumentException("Invalid LAMB checkpoint payload: missing key 'm'");
        }
        restoreSlot(m, mState, "m");

        Object vState = state.get("v");
        if (vState == null) {
            throw new IllegalArgumentException("Invalid LAMB checkpoint payload: missing key 'v'");
        }
        restoreSlot(v, vState, "v");
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
                    "Invalid LAMB checkpoint payload: " + slotName + " must be a List");
        }
        if (serializedSlot.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid LAMB checkpoint payload: " + slotName + " size mismatch (expected "
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
                        "Invalid LAMB checkpoint payload: " + slotName + "[" + i
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
                    "Invalid LAMB checkpoint payload: " + fieldName
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
                            "Invalid LAMB checkpoint payload: " + slotName + "[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid LAMB checkpoint payload: " + slotName + "[" + index + "] must be float[]");
    }
}
