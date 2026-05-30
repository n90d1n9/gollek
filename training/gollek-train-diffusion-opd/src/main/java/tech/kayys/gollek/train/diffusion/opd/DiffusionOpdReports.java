package tech.kayys.gollek.train.diffusion.opd;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.Path;
import java.util.Map;
import tech.kayys.gollek.train.diffusion.api.DiffusionOpdReport;

/**
 * Public loader/parser utilities for normalized DiffusionOPD report artifacts.
 *
 * <p>The selector grammar intentionally supports compact terminal queries so
 * callers can pivot from full row dumps to scalar or summary views such as
 * {@code roundHistory:task=ocr:meanLoss},
 * {@code roundHistory:teacher=ocr-early:lastLoss}, or
 * {@code roundHistory:task=ocr:summary}.
 *
 * <p>{@link DiffusionOpdReportJsons} owns file loading and normalization, while
 * {@link DiffusionOpdReportSelectors} owns selector grammar and round-history
 * projections.
 */
public final class DiffusionOpdReports {
    private DiffusionOpdReports() {
    }

    /**
     * Loads a persisted OPD report artifact from disk into the normalized typed report shape.
     */
    public static DiffusionOpdReport load(Path reportPath) {
        return DiffusionOpdReportJsons.load(reportPath);
    }

    /**
     * Adapts an already-parsed JSON tree into the normalized typed report shape.
     */
    public static DiffusionOpdReport fromJson(JsonNode root) {
        return DiffusionOpdReportJsons.fromJson(root);
    }

    /**
     * Exposes the normalized top-level section map used by the selector grammar and downstream
     * query helpers.
     */
    public static Map<String, Object> sections(DiffusionOpdReport report) {
        return DiffusionOpdReportJsons.sections(report);
    }

    /**
     * Applies the compact selector grammar to a normalized report and returns either a section
     * value, summary view, row list, or scalar projection depending on the selector.
     */
    public static Object select(DiffusionOpdReport report, String section) {
        Map<String, Object> sections = sections(report);
        return DiffusionOpdReportSelectors.select(report, section, sections);
    }
}
