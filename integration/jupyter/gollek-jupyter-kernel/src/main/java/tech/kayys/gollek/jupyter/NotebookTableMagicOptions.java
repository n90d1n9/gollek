package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookMagicArgs.flag;
import static tech.kayys.gollek.jupyter.NotebookMagicArgs.parseLeadingOptions;
import static tech.kayys.gollek.jupyter.NotebookMagicArgs.value;
import static tech.kayys.gollek.jupyter.NotebookTableOps.filterRequiresValue;
import static tech.kayys.gollek.jupyter.NotebookTableOps.normalizeFilterOperator;

import tech.kayys.gollek.jupyter.NotebookMagicArgs.ParsedOptions;

import java.util.List;
import java.util.Locale;

final class NotebookTableMagicOptions {

    private static final int DEFAULT_TABLE_PREVIEW_ROWS = 20;
    private static final int MAX_TABLE_PREVIEW_ROWS = 200;
    private static final int DEFAULT_SAMPLE_ROWS = 5;
    private static final int MAX_SAMPLE_ROWS = 200;
    private static final int DEFAULT_SORT_ROWS = 20;
    private static final int MAX_SORT_ROWS = 200;
    private static final int DEFAULT_FILTER_ROWS = 20;
    private static final int MAX_FILTER_ROWS = 200;
    private static final int DEFAULT_HISTOGRAM_BINS = 10;
    private static final int MAX_HISTOGRAM_BINS = 100;
    private static final int DEFAULT_VALUE_COUNTS_TOP = 20;
    private static final int MAX_VALUE_COUNTS_TOP = 100;

    private NotebookTableMagicOptions() {
    }

    record TableOptions(String path, int previewRows, boolean profile) {}

    record SampleOptions(String path, int rows, Long seed, boolean tsv) {}

    record SortOptions(String path, String column, int rows, boolean descending, boolean tsv) {}

    record FilterOptions(String path, String column, String operator, String value, int rows, boolean tsv) {}

    record ValueCountsOptions(String path, String column, int top, boolean tsv) {}

    record GroupByOptions(String path, String groupColumn, String valueColumn, String aggregate, boolean tsv) {}

    record HistogramOptions(String path, String column, int bins, boolean tsv) {}

    static TableOptions parseTableOptions(String raw, String label) {
        String remaining = raw == null ? "" : raw.trim();
        int previewRows = DEFAULT_TABLE_PREVIEW_ROWS;
        boolean profile = false;
        String magic = "%" + label.toLowerCase(Locale.ROOT);
        while (!remaining.isBlank()) {
            if (remaining.equals("--profile") || remaining.startsWith("--profile ")) {
                profile = true;
                remaining = remaining.substring("--profile".length()).trim();
                continue;
            }
            if (remaining.startsWith("-n ") || remaining.startsWith("--rows ")) {
                String option = remaining.startsWith("-n ") ? "-n" : "--rows";
                String rest = remaining.substring(option.length()).trim();
                int split = rest.indexOf(' ');
                if (split <= 0) {
                    throw new IllegalArgumentException("Usage: " + magic + " [-n N] [--profile] <PATH>");
                }
                String count = rest.substring(0, split).trim();
                try {
                    previewRows = Integer.parseInt(count);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Invalid row count for " + magic + ": " + count);
                }
                remaining = rest.substring(split + 1).trim();
                continue;
            }
            break;
        }
        if (remaining.isBlank()) {
            throw new IllegalArgumentException("Usage: " + magic + " [-n N] [--profile] <PATH>");
        }
        if (previewRows <= 0) {
            throw new IllegalArgumentException("Row count for " + magic + " must be > 0");
        }
        return new TableOptions(remaining, Math.min(previewRows, MAX_TABLE_PREVIEW_ROWS), profile);
    }

    static SampleOptions parseSampleOptions(String raw) {
        String usage = "Usage: %sample [--tsv] [-n N] [--seed S] <PATH>";
        ParsedOptions args = parseLeadingOptions(
                raw,
                usage,
                "Unknown %sample option: ",
                flag("tsv", "--tsv"),
                value("rows", "-n", "--rows"),
                value("seed", "--seed")
        );
        int rows = DEFAULT_SAMPLE_ROWS;
        Long seed = null;
        String rowsText = args.value("rows");
        if (rowsText != null) {
            try {
                rows = Integer.parseInt(rowsText);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid row count for %sample: " + rowsText);
            }
        }
        String seedText = args.value("seed");
        if (seedText != null) {
            try {
                seed = Long.parseLong(seedText);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid seed for %sample: " + seedText);
            }
        }
        if (rows <= 0) {
            throw new IllegalArgumentException("Row count for %sample must be > 0");
        }
        List<String> parts = args.requirePositionals(1, usage);
        return new SampleOptions(parts.getFirst(), Math.min(rows, MAX_SAMPLE_ROWS), seed, args.has("tsv"));
    }

    static SortOptions parseSortOptions(String raw) {
        String usage = "Usage: %sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>";
        ParsedOptions args = parseLeadingOptions(
                raw,
                usage,
                "Unknown %sort option: ",
                flag("tsv", "--tsv"),
                flag("desc", "--desc"),
                flag("asc", "--asc"),
                value("rows", "-n", "--rows")
        );
        int rows = DEFAULT_SORT_ROWS;
        String rowsText = args.value("rows");
        if (rowsText != null) {
            try {
                rows = Integer.parseInt(rowsText);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid row count for %sort: " + rowsText);
            }
        }
        if (rows <= 0) {
            throw new IllegalArgumentException("Row count for %sort must be > 0");
        }
        List<String> parts = args.requirePositionals(2, usage);
        return new SortOptions(
                parts.get(0),
                parts.get(1),
                Math.min(rows, MAX_SORT_ROWS),
                "desc".equals(args.lastPresent("asc", "desc")),
                args.has("tsv")
        );
    }

    static FilterOptions parseFilterOptions(String raw) {
        String usage = "Usage: %filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]";
        ParsedOptions args = parseLeadingOptions(
                raw,
                usage,
                "Unknown %filter option: ",
                flag("tsv", "--tsv"),
                value("rows", "-n", "--rows")
        );
        int rows = DEFAULT_FILTER_ROWS;
        String rowsText = args.value("rows");
        if (rowsText != null) {
            try {
                rows = Integer.parseInt(rowsText);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid row count for %filter: " + rowsText);
            }
        }
        if (rows <= 0) {
            throw new IllegalArgumentException("Row count for %filter must be > 0");
        }
        List<String> parts = args.positionals();
        if (parts.size() < 3 || parts.get(0).isBlank() || parts.get(1).isBlank() || parts.get(2).isBlank()) {
            throw new IllegalArgumentException(usage);
        }
        String operator = normalizeFilterOperator(parts.get(2));
        String value = parts.size() > 3 ? String.join(" ", parts.subList(3, parts.size())).trim() : null;
        if (filterRequiresValue(operator) && (value == null || value.isBlank())) {
            throw new IllegalArgumentException("Operator " + parts.get(2) + " requires VALUE");
        }
        if (!filterRequiresValue(operator)) {
            value = null;
        }
        return new FilterOptions(parts.get(0), parts.get(1), operator, value, Math.min(rows, MAX_FILTER_ROWS), args.has("tsv"));
    }

    static ValueCountsOptions parseValueCountsOptions(String raw) {
        String usage = "Usage: %valuecounts [--tsv] [--top N] <PATH> <COLUMN>";
        ParsedOptions args = parseLeadingOptions(
                raw,
                usage,
                "Unknown %valuecounts option: ",
                flag("tsv", "--tsv"),
                value("top", "--top")
        );
        int top = DEFAULT_VALUE_COUNTS_TOP;
        String topText = args.value("top");
        if (topText != null) {
            try {
                top = Integer.parseInt(topText);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid top count for %valuecounts: " + topText);
            }
        }
        if (top <= 0) {
            throw new IllegalArgumentException("Top count for %valuecounts must be > 0");
        }
        List<String> parts = args.requirePositionals(2, usage);
        return new ValueCountsOptions(parts.get(0), parts.get(1), Math.min(top, MAX_VALUE_COUNTS_TOP), args.has("tsv"));
    }

    static GroupByOptions parseGroupByOptions(String raw) {
        String usage = "Usage: %groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [count|sum|mean|min|max]";
        ParsedOptions args = parseLeadingOptions(
                raw,
                usage,
                "Unknown %groupby option: ",
                flag("tsv", "--tsv")
        );
        List<String> parts = args.requirePositionalsBetween(2, 4, usage);
        String valueColumn = null;
        String aggregate = "count";
        if (parts.size() == 3) {
            if (isGroupAggregate(parts.get(2))) {
                aggregate = parts.get(2).toLowerCase(Locale.ROOT);
                if (!aggregate.equals("count")) {
                    throw new IllegalArgumentException("Aggregate " + aggregate + " requires VALUE_COLUMN");
                }
            } else {
                valueColumn = parts.get(2);
                aggregate = "mean";
            }
        }
        if (parts.size() == 4) {
            valueColumn = parts.get(2);
            aggregate = parts.get(3).toLowerCase(Locale.ROOT);
            if (!isGroupAggregate(aggregate)) {
                throw new IllegalArgumentException("Unknown aggregate for %groupby: " + parts.get(3));
            }
        }
        return new GroupByOptions(parts.get(0), parts.get(1), valueColumn, aggregate, args.has("tsv"));
    }

    static HistogramOptions parseHistogramOptions(String raw) {
        String usage = "Usage: %hist [--tsv] [--bins N] <PATH> <COLUMN>";
        ParsedOptions args = parseLeadingOptions(
                raw,
                usage,
                "Unknown %hist option: ",
                flag("tsv", "--tsv"),
                value("bins", "--bins")
        );
        int bins = DEFAULT_HISTOGRAM_BINS;
        String binsText = args.value("bins");
        if (binsText != null) {
            try {
                bins = Integer.parseInt(binsText);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid bin count for %hist: " + binsText);
            }
        }
        if (bins <= 0) {
            throw new IllegalArgumentException("Bin count for %hist must be > 0");
        }
        List<String> parts = args.requirePositionals(2, usage);
        return new HistogramOptions(parts.get(0), parts.get(1), Math.min(bins, MAX_HISTOGRAM_BINS), args.has("tsv"));
    }

    static boolean isGroupAggregate(String aggregate) {
        return switch (aggregate.toLowerCase(Locale.ROOT)) {
            case "count", "sum", "mean", "min", "max" -> true;
            default -> false;
        };
    }
}
