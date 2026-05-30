package tech.kayys.gollek.train.diffusion.opd;

import java.nio.file.Path;
import java.util.Locale;

/**
 * Owns CLI token parsing for the inspector.
 *
 * <p>This helper stays intentionally narrow: it resolves section/format/output
 * arguments and leaves report loading, section routing, and rendering to the
 * rest of the inspector helper cluster.
 */
final class DiffusionOpdReportInspectorCli {

    private DiffusionOpdReportInspectorCli() {
    }

    static CliOptions parseArgs(String[] args) {
        CliTokens tokens = tokenizeCliArgs(args);
        if (isBundleOutputOnlyInvocation(tokens)) {
            return new CliOptions(tokens.section(), "text", null, Path.of(tokens.third()));
        }
        if (DiffusionOpdReportInspectorSupport.isSupportedFormat(tokens.third())) {
            return parseExplicitFormatArgs(tokens);
        }
        if (tokens.third() != null && inferFormatFromPath(tokens.third()) != null) {
            return new CliOptions(tokens.section(), inferFormatFromPath(tokens.third()), null, Path.of(tokens.third()));
        }
        return parseDefaultArgs(tokens);
    }

    private static CliTokens tokenizeCliArgs(String[] args) {
        return new CliTokens(
                args.length > 1 ? args[1] : "all",
                args.length > 2 ? args[2] : null,
                args.length > 3 ? args[3] : null,
                args.length > 4 ? args[4] : null);
    }

    private static boolean isBundleOutputOnlyInvocation(CliTokens tokens) {
        return tokens.bundleSection()
                && tokens.third() != null
                && !DiffusionOpdReportInspectorSupport.isSupportedFormat(tokens.third())
                && tokens.fourth() == null;
    }

    private static CliOptions parseExplicitFormatArgs(CliTokens tokens) {
        String normalizedFormat = tokens.third().toLowerCase(Locale.ROOT);
        if (tokens.bundleSection() && tokens.fourth() != null && tokens.fifth() == null) {
            return new CliOptions(tokens.section(), normalizedFormat, null, Path.of(tokens.fourth()));
        }
        String columns = DiffusionOpdReportInspectorSupport.normalizeColumnsArgument(tokens.fourth());
        if (columns == null && tokens.fourth() != null && tokens.fifth() == null && looksLikeOutputPath(tokens.fourth())) {
            return new CliOptions(tokens.section(), normalizedFormat, null, Path.of(tokens.fourth()));
        }
        return new CliOptions(
                tokens.section(),
                normalizedFormat,
                columns,
                tokens.fifth() == null ? null : Path.of(tokens.fifth()));
    }

    private static CliOptions parseDefaultArgs(CliTokens tokens) {
        return new CliOptions(
                tokens.section(),
                "text",
                DiffusionOpdReportInspectorSupport.normalizeColumnsArgument(tokens.fourth()),
                tokens.fifth() == null ? null : Path.of(tokens.fifth()));
    }

    private static boolean looksLikeOutputPath(String value) {
        return value != null && inferFormatFromPath(value) != null;
    }

    private static String inferFormatFromPath(String value) {
        if (value == null) {
            return null;
        }
        String normalized = normalizePathToken(value);
        if (matchesAnySuffix(normalized, ".json")) {
            return "json";
        }
        if (matchesAnySuffix(normalized, ".csv")) {
            return "csv";
        }
        if (matchesAnySuffix(normalized, ".table", ".md")) {
            return "table";
        }
        if (matchesAnySuffix(normalized, ".txt", ".log")) {
            return "text";
        }
        return null;
    }

    private static String normalizePathToken(String value) {
        return value.toLowerCase(Locale.ROOT);
    }

    private static boolean matchesAnySuffix(String value, String... suffixes) {
        for (String suffix : suffixes) {
            if (value.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }

    record CliOptions(String section, String format, String columns, Path outputPath) {
    }

    private record CliTokens(String section, String third, String fourth, String fifth) {
        private boolean bundleSection() {
            return section != null && section.startsWith("bundle=");
        }
    }
}
