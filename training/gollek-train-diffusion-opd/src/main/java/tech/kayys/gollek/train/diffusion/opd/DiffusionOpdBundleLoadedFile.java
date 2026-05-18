package tech.kayys.gollek.train.diffusion.opd;

/**
 * Typed loaded bundle file result, carrying both manifest metadata and parsed content.
 */
public record DiffusionOpdBundleLoadedFile(
        String request,
        boolean found,
        DiffusionOpdBundleGeneratedFile file,
        Object content) {
}
