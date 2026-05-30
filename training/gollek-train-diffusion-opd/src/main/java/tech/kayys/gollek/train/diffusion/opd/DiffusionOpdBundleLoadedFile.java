package tech.kayys.gollek.train.diffusion.opd;

/**
 * Typed loaded bundle file result, carrying both manifest metadata and parsed content.
 *
 * <p>This is the file-level counterpart to {@link DiffusionOpdBundleView}: it
 * represents a resolved generated artifact plus its decoded content.
 */
public record DiffusionOpdBundleLoadedFile(
        String request,
        boolean found,
        DiffusionOpdBundleGeneratedFile file,
        Object content) {
}
