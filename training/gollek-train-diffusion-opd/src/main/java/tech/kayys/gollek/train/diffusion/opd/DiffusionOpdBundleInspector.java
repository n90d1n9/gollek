package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;

/**
 * Reusable manifest inspection API for DiffusionOPD bundle artifacts.
 */
public final class DiffusionOpdBundleInspector {

    private DiffusionOpdBundleInspector() {
    }

    public static DiffusionOpdBundleView inspect(
            DiffusionOpdBundleManifest manifest,
            String section,
            String format) {
        return DiffusionOpdReportInspectorSupport.inspectManifestView(manifest, section, format);
    }

    public static DiffusionOpdBundleHealth health(DiffusionOpdBundleManifest manifest) {
        return DiffusionOpdReportInspectorSupport.inspectManifestHealth(manifest);
    }

    public static DiffusionOpdBundleSummary summary(DiffusionOpdBundleManifest manifest) {
        return DiffusionOpdReportInspectorSupport.inspectManifestSummary(manifest);
    }

    public static List<DiffusionOpdBundleGeneratedFile> files(DiffusionOpdBundleManifest manifest) {
        return DiffusionOpdReportInspectorSupport.inspectManifestFiles(manifest);
    }

    public static DiffusionOpdBundleLoadedFile loadFile(
            DiffusionOpdBundleManifest manifest,
            String requested) {
        return DiffusionOpdReportInspectorSupport.inspectManifestLoadedFile(manifest, requested);
    }
}
