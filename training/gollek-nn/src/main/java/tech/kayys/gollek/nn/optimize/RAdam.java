package tech.kayys.gollek.ml.optim;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import tech.kayys.gollek.ml.nn.Parameter;

/**
 * Rectified Adam optimizer.
 *
 * <p>
 * RAdam keeps Adam's adaptive first/second moment tracking, but avoids the
 * high-variance adaptive denominator during early steps. Once the variance term
 * is reliable enough, it switches to the rectified adaptive update.
 */
public final class RAdam implements Optimizer {

    private final List<Parameter> parameters;
    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final Map<Parameter, float[]> m = new HashMap<>();
    private final Map<Parameter, float[]> v = new HashMap<>();
    private int step;

    private RAdam(Builder builder) {
        this.parameters = OptimizerValidation.requireParameters(builder.parameters);
        this.learningRate = OptimizerValidation.learningRate(builder.lr);
        this.beta1 = OptimizerValidation.beta(builder.beta1, "beta1");
        this.beta2 = OptimizerValidation.beta(builder.beta2, "beta2");
        this.epsilon = OptimizerValidation.epsilon(builder.eps);
        this.weightDecay = OptimizerValidation.weightDecay(builder.weightDecay);

        for (Parameter parameter : parameters) {
            int size = (int) parameter.data().numel();
            m.put(parameter, new float[size]);
            v.put(parameter, new float[size]);
        }
    }

    public static Builder builder(List<Parameter> parameters, float lr) {
        return new Builder(parameters, lr);
    }

    @Override
    public void step() {
        OptimizerValidation.requireStepInputs(parameters, "RAdam");
        step++;

        double beta2Power = Math.pow(beta2, step);
        double beta1Correction = 1.0 - Math.pow(beta1, step);
        double rhoInf = 2.0 / (1.0 - beta2) - 1.0;
        double rho = rhoInf - 2.0 * step * beta2Power / (1.0 - beta2Power);
        double rectification = rho > 5.0
                ? Math.sqrt(((rho - 4.0) * (rho - 2.0) * rhoInf)
                        / ((rhoInf - 4.0) * (rhoInf - 2.0) * rho))
                : 1.0;
        double beta2Correction = 1.0 - beta2Power;

        for (Parameter parameter : parameters) {
            if (parameter.grad() == null) {
                continue;
            }

            float[] data = parameter.data().data();
            float[] grad = parameter.grad().data();
            float[] firstMoment = m.get(parameter);
            float[] secondMoment = v.get(parameter);

            for (int i = 0; i < data.length; i++) {
                float g = weightDecay == 0.0f ? grad[i] : grad[i] + weightDecay * data[i];
                firstMoment[i] = beta1 * firstMoment[i] + (1.0f - beta1) * g;
                secondMoment[i] = beta2 * secondMoment[i] + (1.0f - beta2) * g * g;

                double mHat = firstMoment[i] / beta1Correction;
                double update = rho > 5.0
                        ? rectification * mHat / (Math.sqrt(secondMoment[i] / beta2Correction) + epsilon)
                        : mHat;
                data[i] -= (float) (learningRate * update);
            }
        }
    }

    @Override
    public void zeroGrad() {
        for (Parameter parameter : parameters) {
            parameter.zeroGrad();
        }
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void setLearningRate(float lr) {
        this.learningRate = OptimizerValidation.learningRate(lr);
    }

    @Override
    public List<Parameter> parameters() {
        return parameters;
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
        return epsilon;
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
        state.put("optimizer", "RAdam");
        state.put("step", step);
        state.put("learningRate", learningRate);
        state.put("beta1", beta1);
        state.put("beta2", beta2);
        state.put("epsilon", epsilon);
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
        if (optimizerName instanceof String name && !"RAdam".equals(name)) {
            throw new IllegalArgumentException(
                    "Checkpoint optimizer mismatch: expected RAdam but got " + name);
        }

        requireFloatMatch(state.get("beta1"), beta1, "beta1");
        requireFloatMatch(state.get("beta2"), beta2, "beta2");
        requireFloatMatch(state.get("epsilon"), epsilon, "epsilon");
        requireFloatMatch(state.get("weightDecay"), weightDecay, "weightDecay");
        step = Math.max(0, readInt(state.get("step"), step));
        learningRate = OptimizerValidation.learningRate(readFloat(state.get("learningRate"), learningRate));

        Object mState = state.get("m");
        if (mState == null) {
            throw new IllegalArgumentException("Invalid RAdam checkpoint payload: missing key 'm'");
        }
        restoreSlot(m, mState, "m");

        Object vState = state.get("v");
        if (vState == null) {
            throw new IllegalArgumentException("Invalid RAdam checkpoint payload: missing key 'v'");
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
                    "Invalid RAdam checkpoint payload: " + slotName + " must be a List");
        }
        if (serializedSlot.size() != parameters.size()) {
            throw new IllegalArgumentException(
                    "Invalid RAdam checkpoint payload: " + slotName + " size mismatch (expected "
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
                        "Invalid RAdam checkpoint payload: " + slotName + "[" + i
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
                    "Invalid RAdam checkpoint payload: " + fieldName
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
                            "Invalid RAdam checkpoint payload: " + slotName + "[" + index + "][" + i
                                    + "] must be numeric");
                }
                converted[i] = number.floatValue();
            }
            return converted;
        }
        throw new IllegalArgumentException(
                "Invalid RAdam checkpoint payload: " + slotName + "[" + index + "] must be float[]");
    }

    @Override
    public String toString() {
        return String.format("RAdam(lr=%.4f, betas=(%.2f, %.3f), weightDecay=%.4f)",
                learningRate, beta1, beta2, weightDecay);
    }

    public static final class Builder {
        private final List<Parameter> parameters;
        private final float lr;
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float eps = 1.0e-8f;
        private float weightDecay = 0.0f;

        private Builder(List<Parameter> parameters, float lr) {
            this.parameters = parameters;
            this.lr = lr;
        }

        public Builder betas(float b1, float b2) {
            this.beta1 = b1;
            this.beta2 = b2;
            return this;
        }

        public Builder eps(float eps) {
            this.eps = eps;
            return this;
        }

        public Builder weightDecay(float weightDecay) {
            this.weightDecay = weightDecay;
            return this;
        }

        public RAdam build() {
            return new RAdam(this);
        }
    }
}
