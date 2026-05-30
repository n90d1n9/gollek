package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookTables.parseCsvLine;
import static tech.kayys.gollek.jupyter.NotebookTables.parseDelimitedLine;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

final class NotebookDataLoader {

    private NotebookDataLoader() {
    }

    record DelimitedTable(
            Path target,
            String delimiter,
            String label,
            List<String> lines,
            List<String> header,
            List<List<String>> rows
    ) {
        boolean isEmpty() {
            return lines.isEmpty();
        }

        int rowCount() {
            return rows.size();
        }

        int columnCount() {
            return header.size();
        }
    }

    static String regularFileError(Path target) {
        if (!Files.exists(target)) {
            return "File not found: " + target;
        }
        if (!Files.isRegularFile(target)) {
            return "Not a file: " + target;
        }
        return null;
    }

    static DelimitedTable readDelimitedTable(Path target, boolean tsv) throws IOException {
        String delimiter = tsv ? "\t" : ",";
        String label = tsv ? "TSV" : "CSV";
        return readDelimitedTable(target, delimiter, label);
    }

    static DelimitedTable readDelimitedTable(Path target, String delimiter, String label) throws IOException {
        List<String> lines = Files.readAllLines(target, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            return new DelimitedTable(target, delimiter, label, lines, List.of(), List.of());
        }
        List<String> header = ",".equals(delimiter)
                ? parseCsvLine(lines.getFirst())
                : parseDelimitedLine(lines.getFirst(), delimiter);
        List<List<String>> rows = new ArrayList<>();
        for (String line : lines.stream().skip(1).toList()) {
            rows.add(",".equals(delimiter) ? parseCsvLine(line) : parseDelimitedLine(line, delimiter));
        }
        return new DelimitedTable(target, delimiter, label, lines, header, rows);
    }
}
