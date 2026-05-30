package tech.kayys.gollek.train.diffusion.opd;

import java.util.List;

/**
 * Reusable manifest inspection API for DiffusionOPD bundle artifacts.
 *
 * <p>This is the typed public entrypoint for bundle inspection. It keeps
 * callers off the lower-level CLI support surface while delegating manifest
 * section routing, health analysis, and summary shaping to the inspector
 * helper cluster.
 */
public final class DiffusionOpdBundleInspector {

    private DiffusionOpdBundleInspector() {
    }

    /**
     * Inspects one manifest section and returns the section in the requested public view format.
     */
    public static DiffusionOpdBundleView inspect(
            DiffusionOpdBundleManifest manifest,
            String section,
            String format) {
        return DiffusionOpdReportInspectorSupport.inspectManifestView(manifest, section, format);
    }

    /**
     * Computes the richest derived inspection view, including health checks and grouped missing
     * artifact summaries.
     */
    public static DiffusionOpdBundleHealth health(DiffusionOpdBundleManifest manifest) {
        return DiffusionOpdReportInspectorSupport.inspectManifestHealth(manifest);
    }

    /**
     * Returns the lighter-weight composition overview for callers that want summary data without
     * the full health-check payload.
     */
    public static DiffusionOpdBundleSummary summary(DiffusionOpdBundleManifest manifest) {
        return DiffusionOpdReportInspectorSupport.inspectManifestSummary(manifest);
    }

    /**
     * Lists the generated files advertised by the manifest without loading their contents.
     */
    public static List<DiffusionOpdBundleGeneratedFile> files(DiffusionOpdBundleManifest manifest) {
        return DiffusionOpdReportInspectorSupport.inspectManifestFiles(manifest);
    }

    /**
     * Resolves and loads one generated file view, including decoded structured content when
     * supported by the underlying artifact type.
     */
    public static DiffusionOpdBundleLoadedFile loadFile(
            DiffusionOpdBundleManifest manifest,
            String requested) {
        return DiffusionOpdReportInspectorSupport.inspectManifestLoadedFile(manifest, requested);
    }
}
