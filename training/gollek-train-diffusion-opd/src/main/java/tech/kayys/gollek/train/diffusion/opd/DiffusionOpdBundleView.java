package tech.kayys.gollek.train.diffusion.opd;

/**
 * Public typed result of inspecting a bundle manifest section.
 */
public record DiffusionOpdBundleView(String section, String format, Object value) {
}
