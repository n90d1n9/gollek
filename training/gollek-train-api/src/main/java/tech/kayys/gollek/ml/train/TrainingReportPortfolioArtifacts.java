package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;

/**
 * Persists portfolio exports as CI/notebook-friendly JSON and CSV artifacts.
 */
public final class TrainingReportPortfolioArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "portfolio-export.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "portfolio-export.md";
    public static final String DEFAULT_LEADERBOARD_CSV_FILE_NAME = "portfolio-leaderboard.csv";
    public static final String DEFAULT_COMPARISON_METRICS_CSV_FILE_NAME = "portfolio-comparison-metrics.csv";
    public static final String DEFAULT_COMPARISON_FINDINGS_CSV_FILE_NAME = "portfolio-comparison-findings.csv";

    private TrainingReportPortfolioArtifacts() {
    }

    public record Options(
            String jsonFileName,
            String markdownFileName,
            String leaderboardCsvFileName,
            String comparisonMetricsCsvFileName,
            String comparisonFindingsCsvFileName) {
        public Options(
                String jsonFileName,
                String leaderboardCsvFileName,
                String comparisonMetricsCsvFileName,
                String comparisonFindingsCsvFileName) {
            this(
                    jsonFileName,
                    DEFAULT_MARKDOWN_FILE_NAME,
                    leaderboardCsvFileName,
                    comparisonMetricsCsvFileName,
                    comparisonFindingsCsvFileName);
        }

        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(markdownFileName, DEFAULT_MARKDOWN_FILE_NAME, "markdownFileName");
            leaderboardCsvFileName = normalizeFileName(
                    leaderboardCsvFileName,
                    DEFAULT_LEADERBOARD_CSV_FILE_NAME,
                    "leaderboardCsvFileName");
            comparisonMetricsCsvFileName = normalizeFileName(
                    comparisonMetricsCsvFileName,
                    DEFAULT_COMPARISON_METRICS_CSV_FILE_NAME,
                    "comparisonMetricsCsvFileName");
            comparisonFindingsCsvFileName = normalizeFileName(
                    comparisonFindingsCsvFileName,
                    DEFAULT_COMPARISON_FINDINGS_CSV_FILE_NAME,
                    "comparisonFindingsCsvFileName");
        }

        public static Options defaults() {
            return new Options(
                    DEFAULT_JSON_FILE_NAME,
                    DEFAULT_MARKDOWN_FILE_NAME,
                    DEFAULT_LEADERBOARD_CSV_FILE_NAME,
                    DEFAULT_COMPARISON_METRICS_CSV_FILE_NAME,
                    DEFAULT_COMPARISON_FINDINGS_CSV_FILE_NAME);
        }
    }

    public record ArtifactBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path leaderboardCsvFile,
            Path comparisonMetricsCsvFile,
            Path comparisonFindingsCsvFile,
            String jsonSha256,
            String markdownSha256,
            String leaderboardCsvSha256,
            String comparisonMetricsCsvSha256,
            String comparisonFindingsCsvSha256,
            TrainingReportPortfolioExport export) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            leaderboardCsvFile = Objects.requireNonNull(leaderboardCsvFile, "leaderboardCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            comparisonMetricsCsvFile = Objects
                    .requireNonNull(comparisonMetricsCsvFile, "comparisonMetricsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            comparisonFindingsCsvFile = Objects
                    .requireNonNull(comparisonFindingsCsvFile, "comparisonFindingsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(leaderboardCsvSha256, "leaderboardCsvSha256");
            requireChecksum(comparisonMetricsCsvSha256, "comparisonMetricsCsvSha256");
            requireChecksum(comparisonFindingsCsvSha256, "comparisonFindingsCsvSha256");
            export = Objects.requireNonNull(export, "export must not be null");
        }

        public boolean hasComparisons() {
            return export.hasComparisons();
        }

        public boolean hasComparisonFindings() {
            return export.hasComparisonFindings();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("leaderboardCsvFile", leaderboardCsvFile.toString());
            map.put("comparisonMetricsCsvFile", comparisonMetricsCsvFile.toString());
            map.put("comparisonFindingsCsvFile", comparisonFindingsCsvFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("leaderboardCsvSha256", leaderboardCsvSha256);
            map.put("comparisonMetricsCsvSha256", comparisonMetricsCsvSha256);
            map.put("comparisonFindingsCsvSha256", comparisonFindingsCsvSha256);
            map.put("entryCount", export.entryCount());
            map.put("comparisonMetricCount", export.comparisonMetricCount());
            map.put("comparisonFindingCount", export.comparisonFindingCount());
            map.put("hasComparisons", hasComparisons());
            map.put("hasComparisonFindings", hasComparisonFindings());
            map.put("export", export.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path leaderboardCsvFile,
            Path comparisonMetricsCsvFile,
            Path comparisonFindingsCsvFile,
            Map<String, Object> export,
            String markdown,
            String leaderboardCsv,
            String comparisonMetricsCsv,
            String comparisonFindingsCsv,
            String jsonSha256,
            String markdownSha256,
            String leaderboardCsvSha256,
            String comparisonMetricsCsvSha256,
            String comparisonFindingsCsvSha256) {
        public ArtifactInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            leaderboardCsvFile = Objects.requireNonNull(leaderboardCsvFile, "leaderboardCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            comparisonMetricsCsvFile = Objects
                    .requireNonNull(comparisonMetricsCsvFile, "comparisonMetricsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            comparisonFindingsCsvFile = Objects
                    .requireNonNull(comparisonFindingsCsvFile, "comparisonFindingsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            export = immutableStringKeyMap(Objects.requireNonNull(export, "export must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            leaderboardCsv = Objects.requireNonNull(leaderboardCsv, "leaderboardCsv must not be null");
            comparisonMetricsCsv = Objects.requireNonNull(
                    comparisonMetricsCsv,
                    "comparisonMetricsCsv must not be null");
            comparisonFindingsCsv = Objects.requireNonNull(
                    comparisonFindingsCsv,
                    "comparisonFindingsCsv must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(leaderboardCsvSha256, "leaderboardCsvSha256");
            requireChecksum(comparisonMetricsCsvSha256, "comparisonMetricsCsvSha256");
            requireChecksum(comparisonFindingsCsvSha256, "comparisonFindingsCsvSha256");
        }

        public int entryCount() {
            return intValue(export.get("entryCount"), 0);
        }

        public int comparisonMetricCount() {
            return intValue(export.get("comparisonMetricCount"), 0);
        }

        public int comparisonFindingCount() {
            return intValue(export.get("comparisonFindingCount"), 0);
        }

        public boolean hasComparisons() {
            return booleanValue(export.get("hasComparisons"));
        }

        public boolean hasComparisonFindings() {
            return booleanValue(export.get("hasComparisonFindings"));
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("leaderboardCsvFile", leaderboardCsvFile.toString());
            map.put("comparisonMetricsCsvFile", comparisonMetricsCsvFile.toString());
            map.put("comparisonFindingsCsvFile", comparisonFindingsCsvFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("leaderboardCsvSha256", leaderboardCsvSha256);
            map.put("comparisonMetricsCsvSha256", comparisonMetricsCsvSha256);
            map.put("comparisonFindingsCsvSha256", comparisonFindingsCsvSha256);
            map.put("entryCount", entryCount());
            map.put("comparisonMetricCount", comparisonMetricCount());
            map.put("comparisonFindingCount", comparisonFindingCount());
            map.put("hasComparisons", hasComparisons());
            map.put("hasComparisonFindings", hasComparisonFindings());
            map.put("export", export);
            return Map.copyOf(map);
        }
    }

    public record ArtifactVerification(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean leaderboardCsvSha256Matches,
            boolean comparisonMetricsCsvSha256Matches,
            boolean comparisonFindingsCsvSha256Matches,
            boolean markdownMatchesJson,
            boolean leaderboardCsvMatchesJson,
            boolean comparisonMetricsCsvMatchesJson,
            boolean comparisonFindingsCsvMatchesJson,
            List<String> failures) {
        public ArtifactVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedJsonSha256 = normalizeChecksum(expectedJsonSha256);
            expectedMarkdownSha256 = normalizeChecksum(expectedMarkdownSha256);
            expectedLeaderboardCsvSha256 = normalizeChecksum(expectedLeaderboardCsvSha256);
            expectedComparisonMetricsCsvSha256 = normalizeChecksum(expectedComparisonMetricsCsvSha256);
            expectedComparisonFindingsCsvSha256 = normalizeChecksum(expectedComparisonFindingsCsvSha256);
            failures = failures == null ? List.of() : List.copyOf(failures);
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        public void requirePassed() {
            if (!passed()) {
                throw new IllegalStateException(message());
            }
        }

        public String message() {
            if (passed()) {
                return "Portfolio export artifacts verified for " + inspection.directory() + ".";
            }
            return "Portfolio export artifact verification failed: " + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("leaderboardCsvSha256Matches", leaderboardCsvSha256Matches);
            map.put("comparisonMetricsCsvSha256Matches", comparisonMetricsCsvSha256Matches);
            map.put("comparisonFindingsCsvSha256Matches", comparisonFindingsCsvSha256Matches);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("leaderboardCsvMatchesJson", leaderboardCsvMatchesJson);
            map.put("comparisonMetricsCsvMatchesJson", comparisonMetricsCsvMatchesJson);
            map.put("comparisonFindingsCsvMatchesJson", comparisonFindingsCsvMatchesJson);
            map.put("actualJsonSha256", inspection.jsonSha256());
            map.put("actualMarkdownSha256", inspection.markdownSha256());
            map.put("actualLeaderboardCsvSha256", inspection.leaderboardCsvSha256());
            map.put("actualComparisonMetricsCsvSha256", inspection.comparisonMetricsCsvSha256());
            map.put("actualComparisonFindingsCsvSha256", inspection.comparisonFindingsCsvSha256());
            if (expectedJsonSha256 != null) {
                map.put("expectedJsonSha256", expectedJsonSha256);
            }
            if (expectedMarkdownSha256 != null) {
                map.put("expectedMarkdownSha256", expectedMarkdownSha256);
            }
            if (expectedLeaderboardCsvSha256 != null) {
                map.put("expectedLeaderboardCsvSha256", expectedLeaderboardCsvSha256);
            }
            if (expectedComparisonMetricsCsvSha256 != null) {
                map.put("expectedComparisonMetricsCsvSha256", expectedComparisonMetricsCsvSha256);
            }
            if (expectedComparisonFindingsCsvSha256 != null) {
                map.put("expectedComparisonFindingsCsvSha256", expectedComparisonFindingsCsvSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPortfolio portfolio) throws IOException {
        return write(directory, portfolio.export(), Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPortfolio portfolio,
            Options options) throws IOException {
        return write(directory, portfolio.export(), options);
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPortfolio portfolio,
            String baselineName) throws IOException {
        return write(directory, portfolio.exportAgainst(baselineName), Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPortfolio portfolio,
            String baselineName,
            Options options) throws IOException {
        return write(directory, portfolio.exportAgainst(baselineName), options);
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPortfolioExport export) throws IOException {
        return write(directory, export, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportPortfolioExport export,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportPortfolioExport resolvedExport = Objects.requireNonNull(export, "export must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;

        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path leaderboardCsvFile = resolvedDirectory.resolve(resolvedOptions.leaderboardCsvFileName());
        Path comparisonMetricsCsvFile = resolvedDirectory.resolve(resolvedOptions.comparisonMetricsCsvFileName());
        Path comparisonFindingsCsvFile = resolvedDirectory.resolve(resolvedOptions.comparisonFindingsCsvFileName());

        TrainerCheckpointIO.writeStringAtomically(jsonFile, resolvedExport.toJson() + "\n");
        TrainerCheckpointIO.writeStringAtomically(markdownFile, TrainingReportPortfolioMarkdown.render(resolvedExport));
        TrainerCheckpointIO.writeStringAtomically(leaderboardCsvFile, resolvedExport.leaderboardCsv());
        TrainerCheckpointIO.writeStringAtomically(comparisonMetricsCsvFile, resolvedExport.comparisonMetricsCsv());
        TrainerCheckpointIO.writeStringAtomically(comparisonFindingsCsvFile, resolvedExport.comparisonFindingsCsv());

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                leaderboardCsvFile,
                comparisonMetricsCsvFile,
                comparisonFindingsCsvFile,
                TrainerCheckpointIO.sha256Hex(jsonFile),
                TrainerCheckpointIO.sha256Hex(markdownFile),
                TrainerCheckpointIO.sha256Hex(leaderboardCsvFile),
                TrainerCheckpointIO.sha256Hex(comparisonMetricsCsvFile),
                TrainerCheckpointIO.sha256Hex(comparisonFindingsCsvFile),
                resolvedExport);
    }

    public static ArtifactBundle refreshDerived(Path directory) throws IOException {
        return refreshDerived(directory, Options.defaults());
    }

    public static ArtifactBundle refreshDerived(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path leaderboardCsvFile = resolvedDirectory.resolve(resolvedOptions.leaderboardCsvFileName());
        Path comparisonMetricsCsvFile = resolvedDirectory.resolve(resolvedOptions.comparisonMetricsCsvFileName());
        Path comparisonFindingsCsvFile = resolvedDirectory.resolve(resolvedOptions.comparisonFindingsCsvFileName());
        TrainingReportPortfolioExport export = readExport(jsonFile);

        TrainerCheckpointIO.writeStringAtomically(markdownFile, TrainingReportPortfolioMarkdown.render(export));
        TrainerCheckpointIO.writeStringAtomically(leaderboardCsvFile, export.leaderboardCsv());
        TrainerCheckpointIO.writeStringAtomically(comparisonMetricsCsvFile, export.comparisonMetricsCsv());
        TrainerCheckpointIO.writeStringAtomically(comparisonFindingsCsvFile, export.comparisonFindingsCsv());

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                leaderboardCsvFile,
                comparisonMetricsCsvFile,
                comparisonFindingsCsvFile,
                TrainerCheckpointIO.sha256Hex(jsonFile),
                TrainerCheckpointIO.sha256Hex(markdownFile),
                TrainerCheckpointIO.sha256Hex(leaderboardCsvFile),
                TrainerCheckpointIO.sha256Hex(comparisonMetricsCsvFile),
                TrainerCheckpointIO.sha256Hex(comparisonFindingsCsvFile),
                export);
    }

    public static ArtifactInspection refreshDerivedFiles(
            Path jsonFile,
            Path markdownFile,
            Path leaderboardCsvFile,
            Path comparisonMetricsCsvFile,
            Path comparisonFindingsCsvFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedLeaderboardCsvFile = Objects
                .requireNonNull(leaderboardCsvFile, "leaderboardCsvFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedComparisonMetricsCsvFile = Objects
                .requireNonNull(comparisonMetricsCsvFile, "comparisonMetricsCsvFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedComparisonFindingsCsvFile = Objects
                .requireNonNull(comparisonFindingsCsvFile, "comparisonFindingsCsvFile must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportPortfolioExport export = readExport(resolvedJsonFile);

        TrainerCheckpointIO.writeStringAtomically(
                resolvedMarkdownFile,
                TrainingReportPortfolioMarkdown.render(export));
        TrainerCheckpointIO.writeStringAtomically(resolvedLeaderboardCsvFile, export.leaderboardCsv());
        TrainerCheckpointIO.writeStringAtomically(
                resolvedComparisonMetricsCsvFile,
                export.comparisonMetricsCsv());
        TrainerCheckpointIO.writeStringAtomically(
                resolvedComparisonFindingsCsvFile,
                export.comparisonFindingsCsv());

        return readFiles(
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedLeaderboardCsvFile,
                resolvedComparisonMetricsCsvFile,
                resolvedComparisonFindingsCsvFile);
    }

    public static ArtifactInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static ArtifactInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return readFiles(
                resolvedDirectory.resolve(resolvedOptions.jsonFileName()),
                resolvedDirectory.resolve(resolvedOptions.markdownFileName()),
                resolvedDirectory.resolve(resolvedOptions.leaderboardCsvFileName()),
                resolvedDirectory.resolve(resolvedOptions.comparisonMetricsCsvFileName()),
                resolvedDirectory.resolve(resolvedOptions.comparisonFindingsCsvFileName()));
    }

    public static ArtifactInspection readFiles(
            Path jsonFile,
            Path leaderboardCsvFile,
            Path comparisonMetricsCsvFile,
            Path comparisonFindingsCsvFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path parent = resolvedJsonFile.getParent();
        Path markdownFile = parent == null
                ? Path.of(DEFAULT_MARKDOWN_FILE_NAME)
                : parent.resolve(DEFAULT_MARKDOWN_FILE_NAME);
        return readFiles(
                resolvedJsonFile,
                markdownFile,
                leaderboardCsvFile,
                comparisonMetricsCsvFile,
                comparisonFindingsCsvFile);
    }

    public static ArtifactInspection readFiles(
            Path jsonFile,
            Path markdownFile,
            Path leaderboardCsvFile,
            Path comparisonMetricsCsvFile,
            Path comparisonFindingsCsvFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedLeaderboardCsvFile = Objects
                .requireNonNull(leaderboardCsvFile, "leaderboardCsvFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedComparisonMetricsCsvFile = Objects
                .requireNonNull(comparisonMetricsCsvFile, "comparisonMetricsCsvFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedComparisonFindingsCsvFile = Objects
                .requireNonNull(comparisonFindingsCsvFile, "comparisonFindingsCsvFile must not be null")
                .toAbsolutePath()
                .normalize();

        String json = Files.readString(resolvedJsonFile, StandardCharsets.UTF_8);
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        String leaderboardCsv = Files.readString(resolvedLeaderboardCsvFile, StandardCharsets.UTF_8);
        String comparisonMetricsCsv = Files.readString(resolvedComparisonMetricsCsvFile, StandardCharsets.UTF_8);
        String comparisonFindingsCsv = Files.readString(resolvedComparisonFindingsCsvFile, StandardCharsets.UTF_8);
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Portfolio export JSON must be an object: " + resolvedJsonFile);
        }
        return new ArtifactInspection(
                commonDirectory(
                        resolvedJsonFile,
                        resolvedMarkdownFile,
                        resolvedLeaderboardCsvFile,
                        resolvedComparisonMetricsCsvFile,
                        resolvedComparisonFindingsCsvFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedLeaderboardCsvFile,
                resolvedComparisonMetricsCsvFile,
                resolvedComparisonFindingsCsvFile,
                immutableStringKeyMap(map),
                markdown,
                leaderboardCsv,
                comparisonMetricsCsv,
                comparisonFindingsCsv,
                TrainerCheckpointIO.sha256Hex(resolvedJsonFile),
                TrainerCheckpointIO.sha256Hex(resolvedMarkdownFile),
                TrainerCheckpointIO.sha256Hex(resolvedLeaderboardCsvFile),
                TrainerCheckpointIO.sha256Hex(resolvedComparisonMetricsCsvFile),
                TrainerCheckpointIO.sha256Hex(resolvedComparisonFindingsCsvFile));
    }

    public static ArtifactVerification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(
                readFiles(
                        bundle.jsonFile(),
                        bundle.markdownFile(),
                        bundle.leaderboardCsvFile(),
                        bundle.comparisonMetricsCsvFile(),
                        bundle.comparisonFindingsCsvFile()),
                bundle.jsonSha256(),
                bundle.markdownSha256(),
                bundle.leaderboardCsvSha256(),
                bundle.comparisonMetricsCsvSha256(),
                bundle.comparisonFindingsCsvSha256());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256) throws IOException {
        return verify(
                directory,
                expectedJsonSha256,
                null,
                expectedLeaderboardCsvSha256,
                expectedComparisonMetricsCsvSha256,
                expectedComparisonFindingsCsvSha256);
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256) throws IOException {
        return verify(
                read(directory),
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedLeaderboardCsvSha256,
                expectedComparisonMetricsCsvSha256,
                expectedComparisonFindingsCsvSha256);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256) {
        return verify(
                inspection,
                expectedJsonSha256,
                null,
                expectedLeaderboardCsvSha256,
                expectedComparisonMetricsCsvSha256,
                expectedComparisonFindingsCsvSha256);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedLeaderboardCsvSha256,
            String expectedComparisonMetricsCsvSha256,
            String expectedComparisonFindingsCsvSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        String normalizedLeaderboardSha = normalizeChecksum(expectedLeaderboardCsvSha256);
        String normalizedMetricsSha = normalizeChecksum(expectedComparisonMetricsCsvSha256);
        String normalizedFindingsSha = normalizeChecksum(expectedComparisonFindingsCsvSha256);
        boolean jsonMatches = normalizedJsonSha == null
                || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        boolean leaderboardMatches = normalizedLeaderboardSha == null
                || normalizedLeaderboardSha.equalsIgnoreCase(inspection.leaderboardCsvSha256());
        boolean metricsMatches = normalizedMetricsSha == null
                || normalizedMetricsSha.equalsIgnoreCase(inspection.comparisonMetricsCsvSha256());
        boolean findingsMatches = normalizedFindingsSha == null
                || normalizedFindingsSha.equalsIgnoreCase(inspection.comparisonFindingsCsvSha256());
        List<String> failures = new ArrayList<>();
        if (!jsonMatches) {
            failures.add("JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Markdown checksum mismatch for " + inspection.markdownFile());
        }
        if (!leaderboardMatches) {
            failures.add("Leaderboard CSV checksum mismatch for " + inspection.leaderboardCsvFile());
        }
        if (!metricsMatches) {
            failures.add("Comparison metrics CSV checksum mismatch for " + inspection.comparisonMetricsCsvFile());
        }
        if (!findingsMatches) {
            failures.add("Comparison findings CSV checksum mismatch for " + inspection.comparisonFindingsCsvFile());
        }
        TrainingReportPortfolioExport renderedExport = null;
        try {
            renderedExport = TrainingReportPortfolioExport.fromMap(inspection.export());
        } catch (RuntimeException error) {
            failures.add("Portfolio export JSON cannot be rendered for consistency checks: " + error.getMessage());
        }
        boolean markdownMatchesJson = false;
        boolean leaderboardCsvMatchesJson = false;
        boolean comparisonMetricsCsvMatchesJson = false;
        boolean comparisonFindingsCsvMatchesJson = false;
        if (renderedExport != null) {
            markdownMatchesJson = TrainingReportPortfolioMarkdown.render(renderedExport).equals(inspection.markdown());
            leaderboardCsvMatchesJson = renderedExport.leaderboardCsv().equals(inspection.leaderboardCsv());
            comparisonMetricsCsvMatchesJson =
                    renderedExport.comparisonMetricsCsv().equals(inspection.comparisonMetricsCsv());
            comparisonFindingsCsvMatchesJson =
                    renderedExport.comparisonFindingsCsv().equals(inspection.comparisonFindingsCsv());
            if (!markdownMatchesJson) {
                failures.add("Markdown report does not match JSON export: " + inspection.markdownFile());
            }
            if (!leaderboardCsvMatchesJson) {
                failures.add("Leaderboard CSV does not match JSON export: " + inspection.leaderboardCsvFile());
            }
            if (!comparisonMetricsCsvMatchesJson) {
                failures.add("Comparison metrics CSV does not match JSON export: "
                        + inspection.comparisonMetricsCsvFile());
            }
            if (!comparisonFindingsCsvMatchesJson) {
                failures.add("Comparison findings CSV does not match JSON export: "
                        + inspection.comparisonFindingsCsvFile());
            }
        }
        return new ArtifactVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedLeaderboardSha,
                normalizedMetricsSha,
                normalizedFindingsSha,
                jsonMatches,
                markdownMatches,
                leaderboardMatches,
                metricsMatches,
                findingsMatches,
                markdownMatchesJson,
                leaderboardCsvMatchesJson,
                comparisonMetricsCsvMatchesJson,
                comparisonFindingsCsvMatchesJson,
                failures);
    }

    private static String normalizeFileName(String value, String fallback, String fieldName) {
        String normalized = value == null || value.isBlank() ? fallback : value.trim();
        Path path = Path.of(normalized);
        if (path.isAbsolute() || path.getParent() != null || ".".equals(normalized) || "..".equals(normalized)) {
            throw new IllegalArgumentException(fieldName + " must be a file name, not a path: " + value);
        }
        return normalized;
    }

    private static Path commonDirectory(
            Path jsonFile,
            Path markdownFile,
            Path leaderboardCsvFile,
            Path comparisonMetricsCsvFile,
            Path comparisonFindingsCsvFile) {
        Path jsonParent = jsonFile.getParent();
        Path markdownParent = markdownFile.getParent();
        Path leaderboardParent = leaderboardCsvFile.getParent();
        Path metricsParent = comparisonMetricsCsvFile.getParent();
        Path findingsParent = comparisonFindingsCsvFile.getParent();
        if (jsonParent != null
                && jsonParent.equals(markdownParent)
                && jsonParent.equals(leaderboardParent)
                && jsonParent.equals(metricsParent)
                && jsonParent.equals(findingsParent)) {
            return jsonParent;
        }
        return jsonParent == null ? Path.of(".").toAbsolutePath().normalize() : jsonParent;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> source) {
        Object snapshot = TrainerMetadataSupport.immutableSnapshot(source);
        if (snapshot instanceof Map<?, ?> snapshotMap) {
            return (Map<String, Object>) snapshotMap;
        }
        return Map.of();
    }

    private static TrainingReportPortfolioExport readExport(Path jsonFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Portfolio export JSON must be an object: " + resolvedJsonFile);
        }
        try {
            return TrainingReportPortfolioExport.fromMap(immutableStringKeyMap(map));
        } catch (RuntimeException error) {
            throw new IOException("Portfolio export JSON cannot be rendered: " + resolvedJsonFile, error);
        }
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        String normalized = checksum.trim().toLowerCase(java.util.Locale.ROOT);
        if (normalized.length() != 64) {
            throw new IllegalArgumentException("checksum must be a 64-character SHA-256 hex digest");
        }
        for (int index = 0; index < normalized.length(); index++) {
            char character = normalized.charAt(index);
            if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'))) {
                throw new IllegalArgumentException("checksum must be a SHA-256 hex digest");
            }
        }
        return normalized;
    }

    private static void requireChecksum(String checksum, String fieldName) {
        if (checksum == null || checksum.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
    }
}
