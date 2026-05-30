package tech.kayys.gollek.ml.train;

import static tech.kayys.gollek.ml.train.TrainingReportValues.booleanValue;
import static tech.kayys.gollek.ml.train.TrainingReportValues.intValue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import javax.xml.parsers.DocumentBuilderFactory;

/**
 * Persists two-report comparison exports as checksum-verifiable JSON, Markdown,
 * JUnit XML, and CSV artifacts for CI gates and experiment review.
 */
public final class TrainingReportComparisonArtifacts {
    public static final String DEFAULT_JSON_FILE_NAME = "comparison-export.json";
    public static final String DEFAULT_MARKDOWN_FILE_NAME = "comparison-report.md";
    public static final String DEFAULT_JUNIT_XML_FILE_NAME = "comparison-report.junit.xml";
    public static final String DEFAULT_METRICS_CSV_FILE_NAME = "comparison-metrics.csv";
    public static final String DEFAULT_FINDINGS_CSV_FILE_NAME = "comparison-findings.csv";

    private TrainingReportComparisonArtifacts() {
    }

    public record Options(
            String jsonFileName,
            String markdownFileName,
            String junitXmlFileName,
            String metricsCsvFileName,
            String findingsCsvFileName) {
        public Options(
                String jsonFileName,
                String metricsCsvFileName,
                String findingsCsvFileName) {
            this(
                    jsonFileName,
                    DEFAULT_MARKDOWN_FILE_NAME,
                    DEFAULT_JUNIT_XML_FILE_NAME,
                    metricsCsvFileName,
                    findingsCsvFileName);
        }

        public Options(
                String jsonFileName,
                String markdownFileName,
                String metricsCsvFileName,
                String findingsCsvFileName) {
            this(
                    jsonFileName,
                    markdownFileName,
                    DEFAULT_JUNIT_XML_FILE_NAME,
                    metricsCsvFileName,
                    findingsCsvFileName);
        }

        public Options {
            jsonFileName = normalizeFileName(jsonFileName, DEFAULT_JSON_FILE_NAME, "jsonFileName");
            markdownFileName = normalizeFileName(
                    markdownFileName,
                    DEFAULT_MARKDOWN_FILE_NAME,
                    "markdownFileName");
            junitXmlFileName = normalizeFileName(
                    junitXmlFileName,
                    DEFAULT_JUNIT_XML_FILE_NAME,
                    "junitXmlFileName");
            metricsCsvFileName = normalizeFileName(
                    metricsCsvFileName,
                    DEFAULT_METRICS_CSV_FILE_NAME,
                    "metricsCsvFileName");
            findingsCsvFileName = normalizeFileName(
                    findingsCsvFileName,
                    DEFAULT_FINDINGS_CSV_FILE_NAME,
                    "findingsCsvFileName");
        }

        public static Options defaults() {
            return new Options(
                    DEFAULT_JSON_FILE_NAME,
                    DEFAULT_MARKDOWN_FILE_NAME,
                    DEFAULT_JUNIT_XML_FILE_NAME,
                    DEFAULT_METRICS_CSV_FILE_NAME,
                    DEFAULT_FINDINGS_CSV_FILE_NAME);
        }
    }

    public record ArtifactBundle(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path metricsCsvFile,
            Path findingsCsvFile,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256,
            String metricsCsvSha256,
            String findingsCsvSha256,
            TrainingReportComparisonExport export) {
        public ArtifactBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            metricsCsvFile = Objects.requireNonNull(metricsCsvFile, "metricsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            findingsCsvFile = Objects.requireNonNull(findingsCsvFile, "findingsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(junitXmlSha256, "junitXmlSha256");
            requireChecksum(metricsCsvSha256, "metricsCsvSha256");
            requireChecksum(findingsCsvSha256, "findingsCsvSha256");
            export = Objects.requireNonNull(export, "export must not be null");
        }

        public boolean hasFindings() {
            return export.hasFindings();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("junitXmlFile", junitXmlFile.toString());
            map.put("metricsCsvFile", metricsCsvFile.toString());
            map.put("findingsCsvFile", findingsCsvFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("junitXmlSha256", junitXmlSha256);
            map.put("metricsCsvSha256", metricsCsvSha256);
            map.put("findingsCsvSha256", findingsCsvSha256);
            map.put("metricCount", export.metricCount());
            map.put("findingCount", export.findingCount());
            map.put("hasFindings", hasFindings());
            map.put("export", export.toMap());
            return Map.copyOf(map);
        }
    }

    public record ArtifactInspection(
            Path directory,
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path metricsCsvFile,
            Path findingsCsvFile,
            Map<String, Object> export,
            String markdown,
            String junitXml,
            String metricsCsv,
            String findingsCsv,
            String jsonSha256,
            String markdownSha256,
            String junitXmlSha256,
            String metricsCsvSha256,
            String findingsCsvSha256) {
        public ArtifactInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            jsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null").toAbsolutePath().normalize();
            markdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            metricsCsvFile = Objects.requireNonNull(metricsCsvFile, "metricsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            findingsCsvFile = Objects.requireNonNull(findingsCsvFile, "findingsCsvFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            export = immutableStringKeyMap(Objects.requireNonNull(export, "export must not be null"));
            markdown = Objects.requireNonNull(markdown, "markdown must not be null");
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            metricsCsv = Objects.requireNonNull(metricsCsv, "metricsCsv must not be null");
            findingsCsv = Objects.requireNonNull(findingsCsv, "findingsCsv must not be null");
            requireChecksum(jsonSha256, "jsonSha256");
            requireChecksum(markdownSha256, "markdownSha256");
            requireChecksum(junitXmlSha256, "junitXmlSha256");
            requireChecksum(metricsCsvSha256, "metricsCsvSha256");
            requireChecksum(findingsCsvSha256, "findingsCsvSha256");
        }

        public int metricCount() {
            return intValue(export.get("metricCount"), 0);
        }

        public int findingCount() {
            return intValue(export.get("findingCount"), 0);
        }

        public boolean hasFindings() {
            return booleanValue(export.get("hasFindings"));
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("jsonFile", jsonFile.toString());
            map.put("markdownFile", markdownFile.toString());
            map.put("junitXmlFile", junitXmlFile.toString());
            map.put("metricsCsvFile", metricsCsvFile.toString());
            map.put("findingsCsvFile", findingsCsvFile.toString());
            map.put("jsonSha256", jsonSha256);
            map.put("markdownSha256", markdownSha256);
            map.put("junitXmlSha256", junitXmlSha256);
            map.put("metricsCsvSha256", metricsCsvSha256);
            map.put("findingsCsvSha256", findingsCsvSha256);
            map.put("metricCount", metricCount());
            map.put("findingCount", findingCount());
            map.put("hasFindings", hasFindings());
            map.put("export", export);
            return Map.copyOf(map);
        }
    }

    public record ArtifactVerification(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256,
            boolean jsonSha256Matches,
            boolean markdownSha256Matches,
            boolean junitXmlSha256Matches,
            boolean junitXmlWellFormed,
            boolean markdownMatchesJson,
            boolean junitXmlMatchesJson,
            boolean metricsCsvMatchesJson,
            boolean findingsCsvMatchesJson,
            boolean metricsCsvSha256Matches,
            boolean findingsCsvSha256Matches,
            List<String> failures) {
        public ArtifactVerification {
            inspection = Objects.requireNonNull(inspection, "inspection must not be null");
            expectedJsonSha256 = normalizeChecksum(expectedJsonSha256);
            expectedMarkdownSha256 = normalizeChecksum(expectedMarkdownSha256);
            expectedJunitXmlSha256 = normalizeChecksum(expectedJunitXmlSha256);
            expectedMetricsCsvSha256 = normalizeChecksum(expectedMetricsCsvSha256);
            expectedFindingsCsvSha256 = normalizeChecksum(expectedFindingsCsvSha256);
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
                return "Training report comparison artifacts verified for " + inspection.directory() + ".";
            }
            return "Training report comparison artifact verification failed: "
                    + String.join("; ", failures) + ".";
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("jsonSha256Matches", jsonSha256Matches);
            map.put("markdownSha256Matches", markdownSha256Matches);
            map.put("junitXmlSha256Matches", junitXmlSha256Matches);
            map.put("junitXmlWellFormed", junitXmlWellFormed);
            map.put("markdownMatchesJson", markdownMatchesJson);
            map.put("junitXmlMatchesJson", junitXmlMatchesJson);
            map.put("metricsCsvMatchesJson", metricsCsvMatchesJson);
            map.put("findingsCsvMatchesJson", findingsCsvMatchesJson);
            map.put("metricsCsvSha256Matches", metricsCsvSha256Matches);
            map.put("findingsCsvSha256Matches", findingsCsvSha256Matches);
            map.put("actualJsonSha256", inspection.jsonSha256());
            map.put("actualMarkdownSha256", inspection.markdownSha256());
            map.put("actualJunitXmlSha256", inspection.junitXmlSha256());
            map.put("actualMetricsCsvSha256", inspection.metricsCsvSha256());
            map.put("actualFindingsCsvSha256", inspection.findingsCsvSha256());
            if (expectedJsonSha256 != null) {
                map.put("expectedJsonSha256", expectedJsonSha256);
            }
            if (expectedMarkdownSha256 != null) {
                map.put("expectedMarkdownSha256", expectedMarkdownSha256);
            }
            if (expectedJunitXmlSha256 != null) {
                map.put("expectedJunitXmlSha256", expectedJunitXmlSha256);
            }
            if (expectedMetricsCsvSha256 != null) {
                map.put("expectedMetricsCsvSha256", expectedMetricsCsvSha256);
            }
            if (expectedFindingsCsvSha256 != null) {
                map.put("expectedFindingsCsvSha256", expectedFindingsCsvSha256);
            }
            map.put("failures", failures);
            map.put("inspection", inspection.toMap());
            return Map.copyOf(map);
        }
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportComparison comparison) throws IOException {
        return write(directory, requireComparison(comparison).export(), Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportComparison comparison,
            Options options) throws IOException {
        return write(directory, requireComparison(comparison).export(), options);
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportComparisonExport export) throws IOException {
        return write(directory, export, Options.defaults());
    }

    public static ArtifactBundle write(
            Path directory,
            TrainingReportComparisonExport export,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        TrainingReportComparisonExport resolvedExport = Objects.requireNonNull(export, "export must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;

        Path jsonFile = resolvedDirectory.resolve(resolvedOptions.jsonFileName());
        Path markdownFile = resolvedDirectory.resolve(resolvedOptions.markdownFileName());
        Path junitXmlFile = resolvedDirectory.resolve(resolvedOptions.junitXmlFileName());
        Path metricsCsvFile = resolvedDirectory.resolve(resolvedOptions.metricsCsvFileName());
        Path findingsCsvFile = resolvedDirectory.resolve(resolvedOptions.findingsCsvFileName());

        TrainerCheckpointIO.writeStringAtomically(jsonFile, resolvedExport.toJson() + "\n");
        TrainerCheckpointIO.writeStringAtomically(
                markdownFile,
                TrainingReportComparisonMarkdown.render(resolvedExport));
        TrainerCheckpointIO.writeStringAtomically(
                junitXmlFile,
                TrainingReportComparisonJUnitXml.render(resolvedExport));
        TrainerCheckpointIO.writeStringAtomically(metricsCsvFile, resolvedExport.metricsCsv());
        TrainerCheckpointIO.writeStringAtomically(findingsCsvFile, resolvedExport.findingsCsv());

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                junitXmlFile,
                metricsCsvFile,
                findingsCsvFile,
                TrainerCheckpointIO.sha256Hex(jsonFile),
                TrainerCheckpointIO.sha256Hex(markdownFile),
                TrainerCheckpointIO.sha256Hex(junitXmlFile),
                TrainerCheckpointIO.sha256Hex(metricsCsvFile),
                TrainerCheckpointIO.sha256Hex(findingsCsvFile),
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
        Path junitXmlFile = resolvedDirectory.resolve(resolvedOptions.junitXmlFileName());
        Path metricsCsvFile = resolvedDirectory.resolve(resolvedOptions.metricsCsvFileName());
        Path findingsCsvFile = resolvedDirectory.resolve(resolvedOptions.findingsCsvFileName());
        TrainingReportComparisonExport export = readExport(jsonFile);

        TrainerCheckpointIO.writeStringAtomically(markdownFile, TrainingReportComparisonMarkdown.render(export));
        TrainerCheckpointIO.writeStringAtomically(junitXmlFile, TrainingReportComparisonJUnitXml.render(export));
        TrainerCheckpointIO.writeStringAtomically(metricsCsvFile, export.metricsCsv());
        TrainerCheckpointIO.writeStringAtomically(findingsCsvFile, export.findingsCsv());

        return new ArtifactBundle(
                resolvedDirectory,
                jsonFile,
                markdownFile,
                junitXmlFile,
                metricsCsvFile,
                findingsCsvFile,
                TrainerCheckpointIO.sha256Hex(jsonFile),
                TrainerCheckpointIO.sha256Hex(markdownFile),
                TrainerCheckpointIO.sha256Hex(junitXmlFile),
                TrainerCheckpointIO.sha256Hex(metricsCsvFile),
                TrainerCheckpointIO.sha256Hex(findingsCsvFile),
                export);
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
                resolvedDirectory.resolve(resolvedOptions.junitXmlFileName()),
                resolvedDirectory.resolve(resolvedOptions.metricsCsvFileName()),
                resolvedDirectory.resolve(resolvedOptions.findingsCsvFileName()));
    }

    public static ArtifactInspection readFiles(
            Path jsonFile,
            Path markdownFile,
            Path metricsCsvFile,
            Path findingsCsvFile) throws IOException {
        Path parent = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize()
                .getParent();
        Path junitXmlFile = parent == null
                ? Path.of(DEFAULT_JUNIT_XML_FILE_NAME).toAbsolutePath().normalize()
                : parent.resolve(DEFAULT_JUNIT_XML_FILE_NAME);
        return readFiles(jsonFile, markdownFile, junitXmlFile, metricsCsvFile, findingsCsvFile);
    }

    public static ArtifactInspection readFiles(
            Path jsonFile,
            Path markdownFile,
            Path junitXmlFile,
            Path metricsCsvFile,
            Path findingsCsvFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMarkdownFile = Objects.requireNonNull(markdownFile, "markdownFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedMetricsCsvFile = Objects.requireNonNull(metricsCsvFile, "metricsCsvFile must not be null")
                .toAbsolutePath()
                .normalize();
        Path resolvedFindingsCsvFile = Objects.requireNonNull(findingsCsvFile, "findingsCsvFile must not be null")
                .toAbsolutePath()
                .normalize();

        String json = Files.readString(resolvedJsonFile, StandardCharsets.UTF_8);
        String markdown = Files.readString(resolvedMarkdownFile, StandardCharsets.UTF_8);
        String junitXml = Files.readString(resolvedJunitXmlFile, StandardCharsets.UTF_8);
        String metricsCsv = Files.readString(resolvedMetricsCsvFile, StandardCharsets.UTF_8);
        String findingsCsv = Files.readString(resolvedFindingsCsvFile, StandardCharsets.UTF_8);
        Object parsed = TrainerJsonParser.parse(json);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Training report comparison JSON must be an object: " + resolvedJsonFile);
        }
        return new ArtifactInspection(
                commonDirectory(
                        resolvedJsonFile,
                        resolvedMarkdownFile,
                        resolvedJunitXmlFile,
                        resolvedMetricsCsvFile,
                        resolvedFindingsCsvFile),
                resolvedJsonFile,
                resolvedMarkdownFile,
                resolvedJunitXmlFile,
                resolvedMetricsCsvFile,
                resolvedFindingsCsvFile,
                immutableStringKeyMap(map),
                markdown,
                junitXml,
                metricsCsv,
                findingsCsv,
                TrainerCheckpointIO.sha256Hex(resolvedJsonFile),
                TrainerCheckpointIO.sha256Hex(resolvedMarkdownFile),
                TrainerCheckpointIO.sha256Hex(resolvedJunitXmlFile),
                TrainerCheckpointIO.sha256Hex(resolvedMetricsCsvFile),
                TrainerCheckpointIO.sha256Hex(resolvedFindingsCsvFile));
    }

    public static ArtifactVerification verify(ArtifactBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return verify(
                readFiles(
                        bundle.jsonFile(),
                        bundle.markdownFile(),
                        bundle.junitXmlFile(),
                        bundle.metricsCsvFile(),
                        bundle.findingsCsvFile()),
                bundle.jsonSha256(),
                bundle.markdownSha256(),
                bundle.junitXmlSha256(),
                bundle.metricsCsvSha256(),
                bundle.findingsCsvSha256());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) throws IOException {
        return verify(
                directory,
                expectedJsonSha256,
                null,
                null,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256,
                Options.defaults());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) throws IOException {
        return verify(
                directory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                null,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256,
                Options.defaults());
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256,
            Options options) throws IOException {
        return verify(
                read(directory, options),
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256);
    }

    public static ArtifactVerification verify(
            Path directory,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) throws IOException {
        return verify(
                directory,
                expectedJsonSha256,
                expectedMarkdownSha256,
                expectedJunitXmlSha256,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256,
                Options.defaults());
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) {
        return verify(
                inspection,
                expectedJsonSha256,
                null,
                null,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) {
        return verify(
                inspection,
                expectedJsonSha256,
                expectedMarkdownSha256,
                null,
                expectedMetricsCsvSha256,
                expectedFindingsCsvSha256);
    }

    public static ArtifactVerification verify(
            ArtifactInspection inspection,
            String expectedJsonSha256,
            String expectedMarkdownSha256,
            String expectedJunitXmlSha256,
            String expectedMetricsCsvSha256,
            String expectedFindingsCsvSha256) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        String normalizedJsonSha = normalizeChecksum(expectedJsonSha256);
        String normalizedMarkdownSha = normalizeChecksum(expectedMarkdownSha256);
        String normalizedJunitXmlSha = normalizeChecksum(expectedJunitXmlSha256);
        String normalizedMetricsSha = normalizeChecksum(expectedMetricsCsvSha256);
        String normalizedFindingsSha = normalizeChecksum(expectedFindingsCsvSha256);
        boolean jsonMatches = normalizedJsonSha == null
                || normalizedJsonSha.equalsIgnoreCase(inspection.jsonSha256());
        boolean markdownMatches = normalizedMarkdownSha == null
                || normalizedMarkdownSha.equalsIgnoreCase(inspection.markdownSha256());
        boolean junitXmlMatches = normalizedJunitXmlSha == null
                || normalizedJunitXmlSha.equalsIgnoreCase(inspection.junitXmlSha256());
        boolean metricsMatches = normalizedMetricsSha == null
                || normalizedMetricsSha.equalsIgnoreCase(inspection.metricsCsvSha256());
        boolean findingsMatches = normalizedFindingsSha == null
                || normalizedFindingsSha.equalsIgnoreCase(inspection.findingsCsvSha256());
        boolean junitXmlWellFormed = isWellFormedXml(inspection.junitXml());
        TrainingReportComparisonExport renderedExport = null;
        List<String> failures = new ArrayList<>();
        if (!jsonMatches) {
            failures.add("JSON checksum mismatch for " + inspection.jsonFile());
        }
        if (!markdownMatches) {
            failures.add("Markdown checksum mismatch for " + inspection.markdownFile());
        }
        if (!junitXmlMatches) {
            failures.add("JUnit XML checksum mismatch for " + inspection.junitXmlFile());
        }
        if (!junitXmlWellFormed) {
            failures.add("JUnit XML is not well-formed: " + inspection.junitXmlFile());
        }
        try {
            renderedExport = TrainingReportComparisonExport.fromMap(inspection.export());
        } catch (RuntimeException error) {
            failures.add("Comparison export JSON cannot be rendered for consistency checks: " + error.getMessage());
        }
        boolean markdownMatchesJson = false;
        boolean junitXmlMatchesJson = false;
        boolean metricsCsvMatchesJson = false;
        boolean findingsCsvMatchesJson = false;
        if (renderedExport != null) {
            markdownMatchesJson = TrainingReportComparisonMarkdown.render(renderedExport).equals(inspection.markdown());
            junitXmlMatchesJson = TrainingReportComparisonJUnitXml.render(renderedExport).equals(inspection.junitXml());
            metricsCsvMatchesJson = renderedExport.metricsCsv().equals(inspection.metricsCsv());
            findingsCsvMatchesJson = renderedExport.findingsCsv().equals(inspection.findingsCsv());
            if (!markdownMatchesJson) {
                failures.add("Markdown report does not match JSON export: " + inspection.markdownFile());
            }
            if (!junitXmlMatchesJson) {
                failures.add("JUnit XML report does not match JSON export: " + inspection.junitXmlFile());
            }
            if (!metricsCsvMatchesJson) {
                failures.add("Metrics CSV does not match JSON export: " + inspection.metricsCsvFile());
            }
            if (!findingsCsvMatchesJson) {
                failures.add("Findings CSV does not match JSON export: " + inspection.findingsCsvFile());
            }
        }
        if (!metricsMatches) {
            failures.add("Metrics CSV checksum mismatch for " + inspection.metricsCsvFile());
        }
        if (!findingsMatches) {
            failures.add("Findings CSV checksum mismatch for " + inspection.findingsCsvFile());
        }
        return new ArtifactVerification(
                inspection,
                normalizedJsonSha,
                normalizedMarkdownSha,
                normalizedJunitXmlSha,
                normalizedMetricsSha,
                normalizedFindingsSha,
                jsonMatches,
                markdownMatches,
                junitXmlMatches,
                junitXmlWellFormed,
                markdownMatchesJson,
                junitXmlMatchesJson,
                metricsCsvMatchesJson,
                findingsCsvMatchesJson,
                metricsMatches,
                findingsMatches,
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
            Path junitXmlFile,
            Path metricsCsvFile,
            Path findingsCsvFile) {
        Path jsonParent = jsonFile.getParent();
        Path markdownParent = markdownFile.getParent();
        Path junitXmlParent = junitXmlFile.getParent();
        Path metricsParent = metricsCsvFile.getParent();
        Path findingsParent = findingsCsvFile.getParent();
        if (jsonParent != null
                && jsonParent.equals(markdownParent)
                && jsonParent.equals(junitXmlParent)
                && jsonParent.equals(metricsParent)
                && jsonParent.equals(findingsParent)) {
            return jsonParent;
        }
        return jsonParent == null ? Path.of(".").toAbsolutePath().normalize() : jsonParent;
    }

    private static Map<String, Object> immutableStringKeyMap(Map<?, ?> source) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            values.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return Map.copyOf(values);
    }

    private static TrainingReportComparisonExport readExport(Path jsonFile) throws IOException {
        Path resolvedJsonFile = Objects.requireNonNull(jsonFile, "jsonFile must not be null")
                .toAbsolutePath()
                .normalize();
        Object parsed = TrainerJsonParser.parse(Files.readString(resolvedJsonFile, StandardCharsets.UTF_8));
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IOException("Training report comparison JSON must be an object: " + resolvedJsonFile);
        }
        try {
            return TrainingReportComparisonExport.fromMap(immutableStringKeyMap(map));
        } catch (RuntimeException error) {
            throw new IOException("Training report comparison JSON cannot be rendered: " + resolvedJsonFile, error);
        }
    }

    private static TrainingReportComparison requireComparison(TrainingReportComparison comparison) {
        return Objects.requireNonNull(comparison, "comparison must not be null");
    }

    private static void requireChecksum(String checksum, String fieldName) {
        if (checksum == null || checksum.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
    }

    private static String normalizeChecksum(String checksum) {
        if (checksum == null || checksum.isBlank()) {
            return null;
        }
        return checksum.trim();
    }

    private static boolean isWellFormedXml(String xml) {
        return TrainingReportXml.isWellFormed(xml);
    }
}
