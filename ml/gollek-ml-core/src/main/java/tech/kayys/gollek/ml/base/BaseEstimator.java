package tech.kayys.gollek.ml.base;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Shared estimator foundation for the Java ML framework.
 */
public abstract class BaseEstimator implements Cloneable, Serializable {

    private static final long serialVersionUID = 1L;

    private boolean isFitted = false;

    public Map<String, Object> getParams() {
        Map<String, Object> params = new LinkedHashMap<>();

        for (Field field : getClass().getDeclaredFields()) {
            if (isParameterField(field)) {
                field.setAccessible(true);
                try {
                    params.put(field.getName(), field.get(this));
                } catch (IllegalAccessException ignored) {
                }
            }
        }

        return params;
    }

    public BaseEstimator setParams(Map<String, Object> params) {
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            try {
                Field field = getClass().getDeclaredField(entry.getKey());
                field.setAccessible(true);
                field.set(this, entry.getValue());
            } catch (NoSuchFieldException | IllegalAccessException e) {
                throw new IllegalArgumentException("Invalid parameter: " + entry.getKey(), e);
            }
        }
        return this;
    }

    @Override
    public BaseEstimator clone() {
        try {
            BaseEstimator clone = (BaseEstimator) super.clone();
            for (Field field : getClass().getDeclaredFields()) {
                field.setAccessible(true);
                Object value;
                try {
                    value = field.get(this);
                } catch (IllegalAccessException e) {
                    continue;
                }
                if (value instanceof Cloneable) {
                    try {
                        Method cloneMethod = value.getClass().getMethod("clone");
                        field.set(clone, cloneMethod.invoke(value));
                    } catch (Exception ignored) {
                    }
                }
            }
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Failed to clone estimator", e);
        }
    }

    public boolean isFitted() {
        return isFitted;
    }

    protected void setFitted(boolean fitted) {
        this.isFitted = fitted;
    }

    protected void validateData(float[][] x, int[] y) {
        if (x == null || x.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        if (y == null || y.length != x.length) {
            throw new IllegalArgumentException("Labels must match number of samples");
        }
    }

    protected void validateData(float[][] x, int[][] y) {
        if (x == null || x.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        if (y == null || y.length != x.length) {
            throw new IllegalArgumentException("Labels must match number of samples");
        }
    }

    protected void validateInput(float[][] x) {
        if (!isFitted()) {
            throw new IllegalStateException("Estimator must be fitted before prediction");
        }
        if (x == null || x.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
    }

    private boolean isParameterField(Field field) {
        int modifiers = field.getModifiers();
        return !Modifier.isStatic(modifiers)
                && !Modifier.isFinal(modifiers)
                && !Modifier.isTransient(modifiers);
    }

    public abstract void fit(float[][] x, int[] y);

    public void fit(float[][] x, int[][] y) {
        throw new UnsupportedOperationException("This estimator does not support multi-output training");
    }

    public abstract int[] predict(float[][] x);

    public int predictSingle(float[] x) {
        return (int) Math.round(predictSingleValue(x));
    }

    public double[] predictValues(float[][] x) {
        int[] labels = predict(x);
        double[] values = new double[labels.length];
        for (int i = 0; i < labels.length; i++) {
            values[i] = labels[i];
        }
        return values;
    }

    public double predictSingleValue(float[] x) {
        return predictValues(new float[][] { x })[0];
    }

    public double[][] predictProba(float[][] x) {
        throw new UnsupportedOperationException("This estimator does not support probability predictions");
    }

    public double score(float[][] x, int[] y) {
        int[] predictions = predict(x);
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
}
