package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookTables.getCell;
import static tech.kayys.gollek.jupyter.NotebookTables.parseFiniteDouble;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;

final class NotebookStats {

    private NotebookStats() {
    }

    record NumericColumn(String name, int index) {}

    record NumericSummary(
            String name,
            int count,
            int missing,
            double mean,
            double std,
            double min,
            double p25,
            double median,
            double p75,
            double max
    ) {}

    record SchemaColumn(String name, String type, int nonEmpty, int missing, List<String> examples) {}

    record MissingColumn(String name, int missing, int present) {}

    record ColumnProfile(
            String name,
            int nonEmpty,
            int missing,
            int numeric,
            Double min,
            Double max,
            Double mean
    ) {}

    record ValueCount(String value, int count) {}

    record GroupResult(String group, int rows, int numeric, Double sum, Double mean, Double min, Double max) {}

    static final class GroupAccumulator {
        int rows;
        int numeric;
        double sum;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        void add(Double value) {
            rows++;
            if (value == null) {
                return;
            }
            numeric++;
            sum += value;
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        GroupResult toResult(String group) {
            return new GroupResult(
                    group,
                    rows,
                    numeric,
                    numeric == 0 ? null : sum,
                    numeric == 0 ? null : sum / numeric,
                    numeric == 0 ? null : min,
                    numeric == 0 ? null : max
            );
        }
    }

    static List<NumericSummary> describeNumericColumns(List<String> header, List<List<String>> rows) {
        List<NumericSummary> summaries = new ArrayList<>();
        for (int column = 0; column < header.size(); column++) {
            List<Double> values = new ArrayList<>();
            int missing = 0;
            for (List<String> row : rows) {
                String raw = getCell(row, column).trim();
                if (raw.isEmpty()) {
                    missing++;
                    continue;
                }
                try {
                    double value = Double.parseDouble(raw);
                    if (Double.isFinite(value)) {
                        values.add(value);
                    } else {
                        missing++;
                    }
                } catch (NumberFormatException e) {
                    missing++;
                }
            }
            if (values.isEmpty()) {
                continue;
            }
            values.sort(Double::compareTo);
            double sum = values.stream().mapToDouble(Double::doubleValue).sum();
            double mean = sum / values.size();
            double variance = values.stream()
                    .mapToDouble(value -> (value - mean) * (value - mean))
                    .sum() / values.size();
            summaries.add(new NumericSummary(
                    header.get(column),
                    values.size(),
                    missing,
                    mean,
                    Math.sqrt(variance),
                    values.getFirst(),
                    percentile(values, 0.25),
                    percentile(values, 0.5),
                    percentile(values, 0.75),
                    values.getLast()
            ));
        }
        return summaries;
    }

    static List<ColumnProfile> profileColumns(List<String> header, List<List<String>> rows) {
        List<ColumnProfile> profiles = new ArrayList<>();
        for (int column = 0; column < header.size(); column++) {
            int nonEmpty = 0;
            int missing = 0;
            int numeric = 0;
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            double sum = 0.0;
            for (List<String> row : rows) {
                String value = getCell(row, column).trim();
                if (value.isEmpty()) {
                    missing++;
                    continue;
                }
                nonEmpty++;
                Double parsed = parseFiniteDouble(value);
                if (parsed != null) {
                    numeric++;
                    min = Math.min(min, parsed);
                    max = Math.max(max, parsed);
                    sum += parsed;
                }
            }
            profiles.add(new ColumnProfile(
                    header.get(column),
                    nonEmpty,
                    missing,
                    numeric,
                    numeric == 0 ? null : min,
                    numeric == 0 ? null : max,
                    numeric == 0 ? null : sum / numeric
            ));
        }
        return profiles;
    }

    static List<SchemaColumn> inferSchemaColumns(List<String> header, List<List<String>> rows) {
        List<SchemaColumn> columns = new ArrayList<>();
        for (int column = 0; column < header.size(); column++) {
            int nonEmpty = 0;
            int missing = 0;
            int booleanValues = 0;
            int integerValues = 0;
            int decimalValues = 0;
            int dateValues = 0;
            int dateTimeValues = 0;
            LinkedHashSet<String> examples = new LinkedHashSet<>();
            for (List<String> row : rows) {
                String value = getCell(row, column).trim();
                if (value.isEmpty()) {
                    missing++;
                    continue;
                }
                nonEmpty++;
                if (examples.size() < 3) {
                    examples.add(value);
                }
                if (isBooleanLike(value)) {
                    booleanValues++;
                }
                if (isIntegerLike(value)) {
                    integerValues++;
                }
                if (parseFiniteDouble(value) != null) {
                    decimalValues++;
                }
                if (isIsoDate(value)) {
                    dateValues++;
                }
                if (isIsoDateTime(value)) {
                    dateTimeValues++;
                }
            }
            columns.add(new SchemaColumn(
                    header.get(column),
                    inferSchemaType(nonEmpty, booleanValues, integerValues, decimalValues, dateValues, dateTimeValues),
                    nonEmpty,
                    missing,
                    new ArrayList<>(examples)
            ));
        }
        return columns;
    }

    static List<NumericColumn> findNumericColumns(List<String> header, List<List<String>> rows) {
        List<NumericColumn> columns = new ArrayList<>();
        for (int column = 0; column < header.size(); column++) {
            int numeric = 0;
            for (List<String> row : rows) {
                String raw = getCell(row, column).trim();
                if (raw.isEmpty()) {
                    continue;
                }
                try {
                    double value = Double.parseDouble(raw);
                    if (Double.isFinite(value)) {
                        numeric++;
                    }
                } catch (NumberFormatException ignored) {
                }
            }
            if (numeric >= 2) {
                columns.add(new NumericColumn(header.get(column), column));
            }
        }
        return columns;
    }

    static Double correlation(List<List<String>> rows, int leftColumn, int rightColumn) {
        List<Double> leftValues = new ArrayList<>();
        List<Double> rightValues = new ArrayList<>();
        for (List<String> row : rows) {
            Double left = parseFiniteDouble(getCell(row, leftColumn));
            Double right = parseFiniteDouble(getCell(row, rightColumn));
            if (left != null && right != null) {
                leftValues.add(left);
                rightValues.add(right);
            }
        }
        if (leftValues.size() < 2) {
            return null;
        }
        double leftMean = leftValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double rightMean = rightValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double numerator = 0.0;
        double leftDenominator = 0.0;
        double rightDenominator = 0.0;
        for (int i = 0; i < leftValues.size(); i++) {
            double leftDelta = leftValues.get(i) - leftMean;
            double rightDelta = rightValues.get(i) - rightMean;
            numerator += leftDelta * rightDelta;
            leftDenominator += leftDelta * leftDelta;
            rightDenominator += rightDelta * rightDelta;
        }
        double denominator = Math.sqrt(leftDenominator * rightDenominator);
        if (denominator == 0.0) {
            return null;
        }
        return numerator / denominator;
    }

    static Comparator<GroupResult> groupComparator(String aggregate) {
        return (left, right) -> {
            Double leftValue = groupAggregateValue(left, aggregate);
            Double rightValue = groupAggregateValue(right, aggregate);
            int compared = Double.compare(
                    rightValue == null ? Double.NEGATIVE_INFINITY : rightValue,
                    leftValue == null ? Double.NEGATIVE_INFINITY : leftValue
            );
            if (compared != 0) {
                return compared;
            }
            return left.group().compareTo(right.group());
        };
    }

    static Double groupAggregateValue(GroupResult result, String aggregate) {
        return switch (aggregate) {
            case "count" -> (double) result.rows();
            case "sum" -> result.sum();
            case "mean" -> result.mean();
            case "min" -> result.min();
            case "max" -> result.max();
            default -> null;
        };
    }

    private static String inferSchemaType(int nonEmpty, int booleanValues, int integerValues, int decimalValues, int dateValues, int dateTimeValues) {
        if (nonEmpty == 0) {
            return "empty";
        }
        if (booleanValues == nonEmpty) {
            return "boolean";
        }
        if (integerValues == nonEmpty) {
            return "integer";
        }
        if (decimalValues == nonEmpty) {
            return "decimal";
        }
        if (dateValues == nonEmpty) {
            return "date";
        }
        if (dateTimeValues == nonEmpty) {
            return "datetime";
        }
        return "text";
    }

    private static boolean isBooleanLike(String value) {
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        return normalized.equals("true") || normalized.equals("false");
    }

    private static boolean isIntegerLike(String value) {
        return value.trim().matches("[+-]?\\d+");
    }

    private static boolean isIsoDate(String value) {
        try {
            java.time.LocalDate.parse(value.trim());
            return true;
        } catch (java.time.format.DateTimeParseException e) {
            return false;
        }
    }

    private static boolean isIsoDateTime(String value) {
        String trimmed = value.trim();
        try {
            java.time.Instant.parse(trimmed);
            return true;
        } catch (java.time.format.DateTimeParseException ignored) {
        }
        try {
            java.time.OffsetDateTime.parse(trimmed);
            return true;
        } catch (java.time.format.DateTimeParseException ignored) {
        }
        try {
            java.time.LocalDateTime.parse(trimmed);
            return true;
        } catch (java.time.format.DateTimeParseException ignored) {
            return false;
        }
    }

    private static double percentile(List<Double> sortedValues, double percentile) {
        if (sortedValues.size() == 1) {
            return sortedValues.getFirst();
        }
        double position = percentile * (sortedValues.size() - 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        if (lower == upper) {
            return sortedValues.get(lower);
        }
        double weight = position - lower;
        return sortedValues.get(lower) * (1.0 - weight) + sortedValues.get(upper) * weight;
    }
}
