package tech.kayys.gollek.ml;

/**
 * @deprecated Use {@link Gollek} as the unified ML entry point.
 *             {@code GollekML} remains as a thin source-compatible shim for
 *             existing code while all implementation lives in {@link Gollek}.
 */
@Deprecated(since = "0.1.1", forRemoval = true)
public class GollekML {
    public static class DL extends Gollek.DL {
    }

    public static class ML extends Gollek.ML {
    }

    public static class Selection extends Gollek.Selection {
    }

    public static class Hub extends Gollek.Hub {
    }

    public static class Export extends Gollek.Export {
    }
}
