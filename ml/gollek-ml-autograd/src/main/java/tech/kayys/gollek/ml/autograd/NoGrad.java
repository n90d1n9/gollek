package tech.kayys.gollek.ml.autograd;

/**
 * Small try-with-resources guard for disabling gradient tracking in legacy ML APIs.
 */
public final class NoGrad implements AutoCloseable {

    private static final ThreadLocal<Integer> DEPTH = ThreadLocal.withInitial(() -> 0);

    private boolean closed;

    private NoGrad() {
    }

    public static NoGrad enter() {
        DEPTH.set(DEPTH.get() + 1);
        return new NoGrad();
    }

    static boolean active() {
        return DEPTH.get() > 0;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        int depth = DEPTH.get();
        if (depth <= 1) {
            DEPTH.remove();
        } else {
            DEPTH.set(depth - 1);
        }
        closed = true;
    }
}
