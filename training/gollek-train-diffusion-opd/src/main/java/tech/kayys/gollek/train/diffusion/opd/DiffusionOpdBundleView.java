package tech.kayys.gollek.train.diffusion.opd;

/**
 * Public typed result of inspecting a bundle manifest section.
 *
 * <p>Use this when the caller wants an arbitrary manifest section rendered as a
 * typed envelope, rather than one of the higher-level derived views such as
 * {@link DiffusionOpdBundleSummary} or {@link DiffusionOpdBundleHealth}.
 */
public record DiffusionOpdBundleView(String section, String format, Object value) {
}
