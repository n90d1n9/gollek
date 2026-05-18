package tech.kayys.gollek.ml.bytelatent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

final class ByteLatentHistoryCsv {
    private static final String HEADER = "epoch,globalStep,batchCount,trainLoss";

    private ByteLatentHistoryCsv() {
    }

    static void write(Path file, List<ByteLatentHistoryRow> rows) throws IOException {
        List<String> lines = new ArrayList<>(rows.size() + 1);
        lines.add(HEADER);
        for (ByteLatentHistoryRow row : rows) {
            lines.add(row.epoch()
                    + ","
                    + row.globalStep()
                    + ","
                    + row.batchCount()
                    + ","
                    + row.trainLoss());
        }
        Files.write(file, lines, StandardCharsets.UTF_8);
    }

    static List<ByteLatentHistoryRow> read(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            return List.of();
        }
        if (!HEADER.equals(lines.get(0).trim())) {
            throw new IOException("invalid-byte-latent-history-header");
        }
        List<ByteLatentHistoryRow> rows = new ArrayList<>();
        for (int index = 1; index < lines.size(); index++) {
            String line = lines.get(index).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] columns = line.split(",", -1);
            if (columns.length != 4) {
                throw new IOException("invalid-byte-latent-history-row:" + index);
            }
            rows.add(new ByteLatentHistoryRow(
                    Integer.parseInt(columns[0]),
                    Integer.parseInt(columns[1]),
                    Integer.parseInt(columns[2]),
                    Double.parseDouble(columns[3])));
        }
        return List.copyOf(rows);
    }
}
