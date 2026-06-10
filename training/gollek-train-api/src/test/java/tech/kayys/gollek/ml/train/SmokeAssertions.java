package tech.kayys.gollek.ml.train;

final class SmokeAssertions {
    private SmokeAssertions() {
    }

    static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    static void requireEquals(Object expected, Object actual, String label) {
        if (!java.util.Objects.equals(expected, actual)) {
            throw new AssertionError(label + ": expected " + expected + " but got " + actual);
        }
    }

    static void requireNear(double expected, Object actual, String label) {
        if (!(actual instanceof Number number) || Math.abs(number.doubleValue() - expected) > 1.0e-6) {
            throw new AssertionError(label + ": expected " + expected + " but got " + actual);
        }
    }

    static void requireContains(String text, String fragment) {
        if (!text.contains(fragment)) {
            throw new AssertionError("missing fragment: " + fragment + "\n" + text);
        }
    }

    static void requireThrows(Class<? extends Throwable> expected, String message, Runnable action) {
        try {
            action.run();
        } catch (Throwable error) {
            if (!expected.isInstance(error)) {
                throw new AssertionError("expected " + expected.getName() + " but got " + error, error);
            }
            if (message != null && !message.equals(error.getMessage())) {
                throw new AssertionError("expected message " + message + " but got " + error.getMessage(), error);
            }
            return;
        }
        throw new AssertionError("expected " + expected.getName());
    }
}
