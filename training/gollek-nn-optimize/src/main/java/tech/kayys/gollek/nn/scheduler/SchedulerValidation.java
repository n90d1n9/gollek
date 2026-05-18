package tech.kayys.gollek.ml.optim;

final class SchedulerValidation {

    private SchedulerValidation() {
    }

    static float learningRate(float value, String name) {
        return OptimizerValidation.finiteNonNegative(value, name);
    }

    static float positive(float value, String name) {
        return OptimizerValidation.finitePositive(value, name);
    }

    static float nonNegative(float value, String name) {
        return OptimizerValidation.finiteNonNegative(value, name);
    }

    static float factor(float value, String name) {
        if (!Float.isFinite(value) || value <= 0.0f || value >= 1.0f) {
            throw new IllegalArgumentException(name + " must be finite and in (0, 1), got: " + value);
        }
        return value;
    }

    static int readNonNegativeInt(Object value, int fallback, String schedulerName, String fieldName) {
        int result = readInt(value, fallback, schedulerName, fieldName);
        if (result < 0) {
            throw new IllegalArgumentException(
                    "Invalid " + schedulerName + " checkpoint payload: " + fieldName
                            + " must be non-negative, got " + result);
        }
        return result;
    }

    static float readLearningRate(Object value, float fallback, String schedulerName, String fieldName) {
        return learningRate(readFloat(value, fallback, schedulerName, fieldName), fieldName);
    }

    static double readFiniteOrNaNDouble(Object value, double fallback, String schedulerName, String fieldName) {
        double result = readDouble(value, fallback, schedulerName, fieldName);
        if (Double.isInfinite(result)) {
            throw new IllegalArgumentException(
                    "Invalid " + schedulerName + " checkpoint payload: " + fieldName
                            + " must be finite or NaN, got " + result);
        }
        return result;
    }

    static void requireIntMatch(Object value, int expected, String schedulerName, String fieldName) {
        if (value == null) {
            return;
        }
        int loaded = readInt(value, expected, schedulerName, fieldName);
        if (loaded != expected) {
            throw new IllegalArgumentException(
                    "Invalid " + schedulerName + " checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    static void requireFloatMatch(Object value, float expected, String schedulerName, String fieldName) {
        if (value == null) {
            return;
        }
        float loaded = readFloat(value, expected, schedulerName, fieldName);
        if (!Float.isFinite(loaded) || Math.abs(loaded - expected) > 1e-7f) {
            throw new IllegalArgumentException(
                    "Invalid " + schedulerName + " checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    static void requireDoubleMatch(Object value, double expected, String schedulerName, String fieldName) {
        if (value == null) {
            return;
        }
        double loaded = readDouble(value, expected, schedulerName, fieldName);
        if (!Double.isFinite(loaded) || Math.abs(loaded - expected) > 1e-12) {
            throw new IllegalArgumentException(
                    "Invalid " + schedulerName + " checkpoint payload: " + fieldName
                            + " mismatch (expected " + expected + ", got " + loaded + ")");
        }
    }

    private static int readInt(Object value, int fallback, String schedulerName, String fieldName) {
        if (value == null) {
            return fallback;
        }
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value instanceof String text) {
            try {
                return Integer.parseInt(text);
            } catch (NumberFormatException ignored) {
                // fall through to the explicit error below
            }
        }
        throw new IllegalArgumentException(
                "Invalid " + schedulerName + " checkpoint payload: " + fieldName + " must be an integer");
    }

    private static float readFloat(Object value, float fallback, String schedulerName, String fieldName) {
        if (value == null) {
            return fallback;
        }
        if (value instanceof Number number) {
            return number.floatValue();
        }
        if (value instanceof String text) {
            try {
                return Float.parseFloat(text);
            } catch (NumberFormatException ignored) {
                // fall through to the explicit error below
            }
        }
        throw new IllegalArgumentException(
                "Invalid " + schedulerName + " checkpoint payload: " + fieldName + " must be numeric");
    }

    private static double readDouble(Object value, double fallback, String schedulerName, String fieldName) {
        if (value == null) {
            return fallback;
        }
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text) {
            try {
                return Double.parseDouble(text);
            } catch (NumberFormatException ignored) {
                // fall through to the explicit error below
            }
        }
        throw new IllegalArgumentException(
                "Invalid " + schedulerName + " checkpoint payload: " + fieldName + " must be numeric");
    }
}
