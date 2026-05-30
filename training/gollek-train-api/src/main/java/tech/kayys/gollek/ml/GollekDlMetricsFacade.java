package tech.kayys.gollek.ml;

/**
 * Public metric factory facade inherited by {@link Gollek.DL}.
 *
 * <p>The methods intentionally mirror the historical {@code Gollek.DL.*Metric}
 * API so callers keep the same import style while each metric family evolves in
 * its own focused facade.
 */
public class GollekDlMetricsFacade extends GollekDlLanguageModelingMetricsFacade {
    protected GollekDlMetricsFacade() {
    }
}
