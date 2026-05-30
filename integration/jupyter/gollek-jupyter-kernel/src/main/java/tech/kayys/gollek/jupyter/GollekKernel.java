package tech.kayys.gollek.jupyter;

import static tech.kayys.gollek.jupyter.NotebookCharts.buildGroupBySvg;
import static tech.kayys.gollek.jupyter.NotebookCharts.buildHistogramBins;
import static tech.kayys.gollek.jupyter.NotebookCharts.buildMissingSvg;
import static tech.kayys.gollek.jupyter.NotebookCharts.buildValueCountsSvg;
import static tech.kayys.gollek.jupyter.NotebookChartRenderer.histogramPreview;
import static tech.kayys.gollek.jupyter.NotebookChartRenderer.linePlotPreview;
import static tech.kayys.gollek.jupyter.NotebookChartRenderer.scatterPlotPreview;
import static tech.kayys.gollek.jupyter.NotebookStatsRenderer.correlationPreview;
import static tech.kayys.gollek.jupyter.NotebookStatsRenderer.describePreview;
import static tech.kayys.gollek.jupyter.NotebookArchives.listTar;
import static tech.kayys.gollek.jupyter.NotebookArchives.listZip;
import static tech.kayys.gollek.jupyter.NotebookArchives.readArchiveEntryBytes;
import static tech.kayys.gollek.jupyter.NotebookArchives.readTarEntryPreview;
import static tech.kayys.gollek.jupyter.NotebookArchives.readZipEntry;
import static tech.kayys.gollek.jupyter.NotebookArchiveRenderer.archiveEntryPreview;
import static tech.kayys.gollek.jupyter.NotebookArchiveRenderer.archiveListingPreview;
import static tech.kayys.gollek.jupyter.NotebookArchiveRenderer.extractPreview;
import static tech.kayys.gollek.jupyter.NotebookDocuments.prettyPrintJson;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.envFilePreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.htmlPreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.iniPreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.jsonPreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.markdownPreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.propertiesPreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.tomlPreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.xmlPreview;
import static tech.kayys.gollek.jupyter.NotebookDocumentRenderer.yamlPreview;
import static tech.kayys.gollek.jupyter.NotebookDataLoader.readDelimitedTable;
import static tech.kayys.gollek.jupyter.NotebookDataLoader.regularFileError;
import static tech.kayys.gollek.jupyter.NotebookDependencies.classpathEntries;
import static tech.kayys.gollek.jupyter.NotebookDependencies.fetchMavenArtifactError;
import static tech.kayys.gollek.jupyter.NotebookDependencies.parseMavenMagicArgs;
import static tech.kayys.gollek.jupyter.NotebookDependencies.resolveMavenArtifact;
import static tech.kayys.gollek.jupyter.NotebookDependencies.validateJarPath;
import static tech.kayys.gollek.jupyter.NotebookDependencyRenderer.classpathPreview;
import static tech.kayys.gollek.jupyter.NotebookDependencyRenderer.dependenciesPreview;
import static tech.kayys.gollek.jupyter.NotebookDependencyRenderer.jarAddedPreview;
import static tech.kayys.gollek.jupyter.NotebookDependencyRenderer.localMavenArtifactAddedPreview;
import static tech.kayys.gollek.jupyter.NotebookDependencyRenderer.missingMavenArtifactPreview;
import static tech.kayys.gollek.jupyter.NotebookDependencyRenderer.remoteMavenArtifactAddedPreview;
import static tech.kayys.gollek.jupyter.NotebookFiles.collectGrepMatches;
import static tech.kayys.gollek.jupyter.NotebookFiles.computeDiskUsage;
import static tech.kayys.gollek.jupyter.NotebookFiles.formatPathEntry;
import static tech.kayys.gollek.jupyter.NotebookFiles.readUtf8Preview;
import static tech.kayys.gollek.jupyter.NotebookFileMagicOptions.parseExtractOptions;
import static tech.kayys.gollek.jupyter.NotebookFileMagicOptions.parseLinePreviewOptions;
import static tech.kayys.gollek.jupyter.NotebookFileMagicOptions.parseSearchOptions;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.diffPreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.directoryPreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.diskUsagePreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.filePreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.linePreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.matchesPreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.sha256Preview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.treePreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.wordCountPreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.workingDirectoryChangedPreview;
import static tech.kayys.gollek.jupyter.NotebookFileRenderer.workingDirectoryPreview;
import static tech.kayys.gollek.jupyter.NotebookRuntimeRenderer.classLocationPreview;
import static tech.kayys.gollek.jupyter.NotebookRuntimeRenderer.envPreview;
import static tech.kayys.gollek.jupyter.NotebookRuntimeRenderer.resetPreview;
import static tech.kayys.gollek.jupyter.NotebookRuntimeRenderer.timedResultDisplay;
import static tech.kayys.gollek.jupyter.NotebookRuntimeMagicOptions.parseTimeitOptions;
import static tech.kayys.gollek.jupyter.NotebookTableOps.compareSortRows;
import static tech.kayys.gollek.jupyter.NotebookTableOps.filterPredicateLabel;
import static tech.kayys.gollek.jupyter.NotebookTableOps.isNumericSortColumn;
import static tech.kayys.gollek.jupyter.NotebookTableOps.matchesFilter;
import static tech.kayys.gollek.jupyter.NotebookTableOps.sampleRows;
import static tech.kayys.gollek.jupyter.NotebookTableMagicOptions.parseFilterOptions;
import static tech.kayys.gollek.jupyter.NotebookTableMagicOptions.parseGroupByOptions;
import static tech.kayys.gollek.jupyter.NotebookTableMagicOptions.parseHistogramOptions;
import static tech.kayys.gollek.jupyter.NotebookTableMagicOptions.parseSampleOptions;
import static tech.kayys.gollek.jupyter.NotebookTableMagicOptions.parseSortOptions;
import static tech.kayys.gollek.jupyter.NotebookTableMagicOptions.parseTableOptions;
import static tech.kayys.gollek.jupyter.NotebookTableMagicOptions.parseValueCountsOptions;
import static tech.kayys.gollek.jupyter.NotebookStats.correlation;
import static tech.kayys.gollek.jupyter.NotebookStats.describeNumericColumns;
import static tech.kayys.gollek.jupyter.NotebookStats.findNumericColumns;
import static tech.kayys.gollek.jupyter.NotebookStats.groupComparator;
import static tech.kayys.gollek.jupyter.NotebookStats.inferSchemaColumns;
import static tech.kayys.gollek.jupyter.NotebookStats.profileColumns;
import static tech.kayys.gollek.jupyter.NotebookTables.formatNullableStat;
import static tech.kayys.gollek.jupyter.NotebookTables.getCell;
import static tech.kayys.gollek.jupyter.NotebookTables.parseFiniteDouble;
import static tech.kayys.gollek.jupyter.NotebookTableRenderer.delimitedTablePreview;
import static tech.kayys.gollek.jupyter.NotebookTableRenderer.filterPreview;
import static tech.kayys.gollek.jupyter.NotebookTableRenderer.htmlRows;
import static tech.kayys.gollek.jupyter.NotebookTableRenderer.plainRows;
import static tech.kayys.gollek.jupyter.NotebookTableRenderer.samplePreview;
import static tech.kayys.gollek.jupyter.NotebookTableRenderer.sortPreview;
import static tech.kayys.gollek.jupyter.NotebookTableRenderer.scrollableHtmlRows;
import static tech.kayys.gollek.jupyter.NotebookSummaryRenderer.groupByPreview;
import static tech.kayys.gollek.jupyter.NotebookSummaryRenderer.missingPreview;
import static tech.kayys.gollek.jupyter.NotebookSummaryRenderer.schemaPreview;
import static tech.kayys.gollek.jupyter.NotebookSummaryRenderer.valueCountsPreview;

import tech.kayys.gollek.jupyter.NotebookCharts.HistogramBin;
import tech.kayys.gollek.jupyter.NotebookCharts.LinePlotPoint;
import tech.kayys.gollek.jupyter.NotebookCharts.ScatterPoint;
import tech.kayys.gollek.jupyter.NotebookArchives.ArchiveEntry;
import tech.kayys.gollek.jupyter.NotebookArchives.ArchiveListing;
import tech.kayys.gollek.jupyter.NotebookFileMagicOptions.ExtractOptions;
import tech.kayys.gollek.jupyter.NotebookFileMagicOptions.LinePreviewOptions;
import tech.kayys.gollek.jupyter.NotebookFileMagicOptions.SearchOptions;
import tech.kayys.gollek.jupyter.NotebookFiles.DiskUsageStats;
import tech.kayys.gollek.jupyter.NotebookFiles.TextPreview;
import tech.kayys.gollek.jupyter.NotebookDataLoader.DelimitedTable;
import tech.kayys.gollek.jupyter.NotebookDependencies.MavenMagicArgs;
import tech.kayys.gollek.jupyter.NotebookDependencies.MavenLookup;
import tech.kayys.gollek.jupyter.NotebookRuntimeMagicOptions.TimeitOptions;
import tech.kayys.gollek.jupyter.NotebookStats.GroupAccumulator;
import tech.kayys.gollek.jupyter.NotebookStats.GroupResult;
import tech.kayys.gollek.jupyter.NotebookStats.ColumnProfile;
import tech.kayys.gollek.jupyter.NotebookStats.MissingColumn;
import tech.kayys.gollek.jupyter.NotebookStats.NumericColumn;
import tech.kayys.gollek.jupyter.NotebookStats.NumericSummary;
import tech.kayys.gollek.jupyter.NotebookStats.SchemaColumn;
import tech.kayys.gollek.jupyter.NotebookStats.ValueCount;
import tech.kayys.gollek.jupyter.NotebookTableOps.SortRow;
import tech.kayys.gollek.jupyter.NotebookTableMagicOptions.FilterOptions;
import tech.kayys.gollek.jupyter.NotebookTableMagicOptions.GroupByOptions;
import tech.kayys.gollek.jupyter.NotebookTableMagicOptions.HistogramOptions;
import tech.kayys.gollek.jupyter.NotebookTableMagicOptions.SampleOptions;
import tech.kayys.gollek.jupyter.NotebookTableMagicOptions.SortOptions;
import tech.kayys.gollek.jupyter.NotebookTableMagicOptions.TableOptions;
import tech.kayys.gollek.jupyter.NotebookTableMagicOptions.ValueCountsOptions;

import org.dflib.jjava.execution.CodeEvaluator;
import org.dflib.jjava.execution.CodeEvaluatorBuilder;
import org.dflib.jjava.jupyter.kernel.BaseKernel;
import org.dflib.jjava.jupyter.kernel.LanguageInfo;
import org.dflib.jjava.jupyter.kernel.ReplacementOptions;
import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Stream;

public class GollekKernel extends BaseKernel {

    private static final int MAX_CAT_BYTES = 32 * 1024;
    private static final int MAX_DU_ENTRIES = 2_000;
    private static final int MAX_MISSING_CHART_COLUMNS = 100;
    private static final int MAX_GROUPBY_GROUPS = 100;
    private static final long MAX_EXTRACT_BYTES = 64L * 1024 * 1024;

    private static final LanguageInfo LANGUAGE_INFO = new LanguageInfo.Builder("java")
            .version(Runtime.version().toString())
            .mimetype("text/x-java-source")
            .fileExtension(".jshell")
            .pygments("java")
            .codemirror("java")
            .build();

    private CodeEvaluator evaluator;
    private CompletionProvider completer;
    private final Set<String> dynamicClasspathEntries;
    private final NotebookMagicDispatcher magicDispatcher;
    private Path currentDirectory;

    public GollekKernel() {
        this.dynamicClasspathEntries = new LinkedHashSet<>();
        this.magicDispatcher = createMagicDispatcher();
        this.currentDirectory = Path.of("").toAbsolutePath().normalize();
        reinitializeEvaluator();
    }

    @Override
    public DisplayData eval(String code) throws Exception {
        DisplayData magic = evalMagic(code);
        if (magic != null) {
            return magic;
        }
        Object result = evaluator.eval(code);
        if (result == null) return DisplayData.EMPTY;
        DisplayData rich = TensorDisplay.render(result, getRenderer());
        return rich != null ? rich : new DisplayData(result.toString());
    }

    @Override
    public ReplacementOptions complete(String code, int cursor) throws Exception {
        return completer.complete(code, cursor);
    }

    @Override
    public String isComplete(String code) { return evaluator.isComplete(code); }

    @Override
    public void interrupt() { evaluator.interrupt(); }

    @Override
    public void onShutdown(boolean restart) { evaluator.shutdown(); }

    @Override
    public LanguageInfo getLanguageInfo() { return LANGUAGE_INFO; }

    @Override
    public String getBanner() {
        return "Gollek Kernel — Java " + Runtime.version()
                + "\nGollek ML libs on classpath"
                + "\nPreloaded: GradTensor, Linear, ReLU, Sequential"
                + "\nHelpers: gollekHelp(), gollekTensorDemo(), gollekPreviewTensorDemo(), gollekMetricsDemo(), gollekLossCurveDemo(), gollekImageDemo(), gollekAudioDemo(), gollekModelDemo(), gollekModelSummary()"
                + "\n" + NotebookMagicCatalog.bannerLine();
    }

    @Override
    public List<LanguageInfo.Help> getHelpLinks() {
        return List.of(new LanguageInfo.Help("Gollek Docs", "https://gollek-ai.github.io/docs/"));
    }

    private DisplayData evalMagic(String code) throws Exception {
        return magicDispatcher.evaluate(code);
    }

    private static DisplayData displayPreview(NotebookPreview preview) {
        DisplayData data = new DisplayData(preview.plain());
        data.putData(MIMEType.TEXT_HTML, preview.html());
        return data;
    }

    private NotebookMagicDispatcher createMagicDispatcher() {
        return new NotebookMagicDispatcher()
                .exact("%classpath", ignored -> renderClasspath())
                .exact("%magics", ignored -> NotebookMagicCatalog.renderMagics())
                .exact("%reset", ignored -> renderReset())
                .exact("%pwd", ignored -> renderPwd())
                .exact("%ls", ignored -> renderLs())
                .exact("%tree", ignored -> renderTree(null))
                .exact("%du", ignored -> renderDiskUsage(null))
                .exact("%deps", ignored -> renderDeps())
                .prefix("%env", this::renderEnv)
                .prefix("%tree", this::renderTree)
                .prefix("%du", this::renderDiskUsage)
                .prefix("%cat", this::renderCat)
                .prefix("%head", this::renderHead)
                .prefix("%tail", this::renderTail)
                .prefix("%grep", this::renderGrep)
                .prefix("%findfile", this::renderFindFile)
                .prefix("%wc", this::renderWordCount)
                .prefix("%diff", this::renderDiff)
                .prefix("%zipls", this::renderZipList)
                .prefix("%zipcat", this::renderZipCat)
                .prefix("%tarls", this::renderTarList)
                .prefix("%tarcat", this::renderTarCat)
                .prefix("%extract", this::renderExtract)
                .prefix("%sha256", this::renderSha256)
                .prefix("%json", this::renderJson)
                .prefix("%csv", this::renderCsv)
                .prefix("%tsv", raw -> renderDelimitedTable(raw, "\t", "TSV"))
                .prefix("%sample", this::renderSample)
                .prefix("%sort", this::renderSort)
                .prefix("%filter", this::renderFilter)
                .prefix("%valuecounts", this::renderValueCounts)
                .prefix("%schema", this::renderSchema)
                .prefix("%missing", this::renderMissing)
                .prefix("%groupby", this::renderGroupBy)
                .prefix("%lineplot", this::renderLinePlot)
                .prefix("%scatter", this::renderScatterPlot)
                .prefix("%hist", this::renderHistogram)
                .prefix("%describe", this::renderDescribe)
                .prefix("%corr", this::renderCorrelation)
                .prefix("%md", this::renderMarkdown)
                .prefix("%html", this::renderHtmlFile)
                .prefix("%yaml", this::renderYamlFile)
                .prefix("%toml", this::renderTomlFile)
                .prefix("%xml", this::renderXmlFile)
                .prefix("%ini", this::renderIniFile)
                .prefix("%properties", this::renderPropertiesFile)
                .prefix("%envfile", this::renderEnvFile)
                .prefix("%cd", this::renderCd)
                .prefix("%help", NotebookMagicCatalog::renderHelp)
                .prefix("%addjar", this::renderAddJar)
                .prefix("%maven", this::renderMaven)
                .prefix("%whichclass", this::renderWhichClass)
                .prefix("%timeit", this::renderTimeit)
                .prefix("%time", this::renderTimedEval);
    }

    private void reinitializeEvaluator() {
        this.evaluator = new CodeEvaluatorBuilder()
                .sysStdout().sysStderr().sysStdin()
                .compilerOpts(
                        "--enable-preview",
                        "--source", Integer.toString(Runtime.version().feature()),
                        "--add-modules=jdk.incubator.vector"
                )
                .startupScript(NotebookStartupScripts.defaultScript())
                .build();
        this.completer = new CompletionProvider(evaluator.getShell());
    }

    private DisplayData renderReset() {
        evaluator.shutdown();
        dynamicClasspathEntries.clear();
        reinitializeEvaluator();

        return displayPreview(resetPreview());
    }

    private DisplayData renderTimedEval(String expr) throws Exception {
        long started = System.nanoTime();
        Object result = evaluator.eval(expr);
        double elapsedMs = (System.nanoTime() - started) / 1_000_000.0;
        String timing = String.format(java.util.Locale.ROOT, "elapsedMs=%.3f", elapsedMs);

        DisplayData rich = TensorDisplay.render(result, getRenderer());
        return timedResultDisplay(timing, result, rich);
    }

    private DisplayData renderTimeit(String raw) throws Exception {
        TimeitOptions options;
        try {
            options = parseTimeitOptions(raw);
        } catch (IllegalArgumentException e) {
            return new DisplayData(e.getMessage());
        }

        double minMs = Double.POSITIVE_INFINITY;
        double maxMs = Double.NEGATIVE_INFINITY;
        double totalMs = 0.0;
        Object lastResult = null;
        for (int i = 0; i < options.runs(); i++) {
            long started = System.nanoTime();
            lastResult = evaluator.eval(options.expression());
            double elapsedMs = (System.nanoTime() - started) / 1_000_000.0;
            minMs = Math.min(minMs, elapsedMs);
            maxMs = Math.max(maxMs, elapsedMs);
            totalMs += elapsedMs;
        }
        double avgMs = totalMs / options.runs();
        String summary = String.format(Locale.ROOT, "runs=%d avgMs=%.3f minMs=%.3f maxMs=%.3f", options.runs(), avgMs, minMs, maxMs);
        DisplayData rich = TensorDisplay.render(lastResult, getRenderer());
        return timedResultDisplay(summary, lastResult, rich);
    }

    private DisplayData renderClasspath() {
        List<String> nonBlankEntries = classpathEntries(dynamicClasspathEntries);
        return displayPreview(classpathPreview(nonBlankEntries));
    }

    private DisplayData renderDeps() {
        List<String> entries = dynamicClasspathEntries.stream().toList();
        return displayPreview(dependenciesPreview(entries));
    }

    private DisplayData renderAddJar(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %addjar </absolute/or/relative/path/to/file.jar>");
        }
        try {
            Path jar = Path.of(rawPath).toAbsolutePath().normalize();
            String validation = validateJarPath(jar);
            if (validation != null) {
                return new DisplayData(validation);
            }
            addJarToClasspath(jar);
            return displayPreview(jarAddedPreview(jar.toString()));
        } catch (Exception e) {
            return new DisplayData("Unable to add jar: " + e.getMessage());
        }
    }

    private DisplayData renderMaven(String coords) {
        if (coords == null || coords.isBlank()) {
            return new DisplayData("Usage: %maven [--allow-remote] [--fetch] [--explain] <groupId:artifactId:version>");
        }
        MavenMagicArgs args = parseMavenMagicArgs(coords);
        if (!args.valid()) {
            return new DisplayData(args.error());
        }
        String normalized = args.coordinates();
        MavenLookup lookup = resolveMavenArtifact(args.groupId(), args.artifactId(), args.version());
        if (lookup.jar() == null) {
            if (args.allowRemote() && args.fetch()) {
                String fetchError = fetchMavenArtifactError(normalized);
                if (fetchError != null) {
                    return new DisplayData(fetchError);
                }
                lookup = resolveMavenArtifact(args.groupId(), args.artifactId(), args.version());
                if (lookup.jar() != null) {
                    try {
                        addJarToClasspath(lookup.jar());
                        return displayPreview(remoteMavenArtifactAddedPreview(normalized, lookup.jar().toString()));
                    } catch (Exception e) {
                        return new DisplayData("Fetched artifact but could not add it: " + e.getMessage());
                    }
                }
            }
            return displayPreview(missingMavenArtifactPreview(normalized, lookup, args.allowRemote(), args.explain()));
        }
        try {
            addJarToClasspath(lookup.jar());
            return displayPreview(localMavenArtifactAddedPreview(normalized, lookup.jar().toString()));
        } catch (Exception e) {
            return new DisplayData("Unable to add Maven artifact: " + e.getMessage());
        }
    }

    private void addJarToClasspath(Path jar) {
        evaluator.getShell().addToClasspath(jar.toString());
        dynamicClasspathEntries.add(jar.toString());
    }

    private DisplayData renderPwd() {
        String pwd = currentDirectory.toString();
        return displayPreview(workingDirectoryPreview(pwd));
    }

    private DisplayData renderCd(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %cd <PATH>");
        }
        try {
            Path next = currentDirectory.resolve(rawPath).normalize();
            if (Path.of(rawPath).isAbsolute()) {
                next = Path.of(rawPath).normalize();
            }
            if (!Files.exists(next)) {
                return new DisplayData("Directory not found: " + next);
            }
            if (!Files.isDirectory(next)) {
                return new DisplayData("Not a directory: " + next);
            }
            currentDirectory = next.toAbsolutePath().normalize();
            System.setProperty("user.dir", currentDirectory.toString());
            return displayPreview(workingDirectoryChangedPreview(currentDirectory.toString()));
        } catch (Exception e) {
            return new DisplayData("Unable to change directory: " + e.getMessage());
        }
    }

    private DisplayData renderLs() {
        Path cwd = currentDirectory;
        try (Stream<Path> stream = Files.list(cwd)) {
            List<Path> entries = stream
                    .sorted(Comparator.comparing(path -> path.getFileName().toString().toLowerCase(Locale.ROOT)))
                    .limit(50)
                    .toList();
            List<String> labels = entries.stream()
                    .map(entry -> formatPathEntry(entry))
                    .toList();
            return displayPreview(directoryPreview(cwd.toString(), labels));
        } catch (Exception e) {
            return new DisplayData("Unable to list directory: " + e.getMessage());
        }
    }

    private DisplayData renderTree(String rawPath) {
        Path root = currentDirectory;
        try {
            if (rawPath != null && !rawPath.isBlank()) {
                root = resolveNotebookPath(rawPath);
            }
            root = root.toAbsolutePath().normalize();
            final Path treeRoot = root;
            if (!Files.exists(treeRoot)) {
                return new DisplayData("Tree target not found: " + treeRoot);
            }
            if (!Files.isDirectory(treeRoot)) {
                return new DisplayData("Tree target is not a directory: " + treeRoot);
            }

            List<String> lines = new java.util.ArrayList<>();
            lines.add("Tree(" + treeRoot + ")");
            try (Stream<Path> stream = Files.walk(treeRoot, 3)) {
                stream
                        .filter(path -> !path.equals(treeRoot))
                        .sorted(Comparator.comparing(Path::toString, String.CASE_INSENSITIVE_ORDER))
                        .limit(60)
                        .forEach(path -> {
                            Path relative = treeRoot.relativize(path);
                            int depth = relative.getNameCount();
                            String indent = "  ".repeat(Math.max(0, depth - 1));
                            lines.add(indent + formatPathEntry(path));
                        });
            }

            return displayPreview(treePreview(treeRoot.toString(), lines));
        } catch (Exception e) {
            return new DisplayData("Unable to render tree: " + e.getMessage());
        }
    }

    private DisplayData renderDiskUsage(String rawPath) {
        try {
            Path target = (rawPath == null || rawPath.isBlank())
                    ? currentDirectory
                    : resolveNotebookPath(rawPath);
            target = target.toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("DU target not found: " + target);
            }
            if (!Files.isRegularFile(target) && !Files.isDirectory(target)) {
                return new DisplayData("DU target is not a file or directory: " + target);
            }

            DiskUsageStats stats = computeDiskUsage(target, MAX_DU_ENTRIES);
            String type = Files.isDirectory(target) ? "directory" : "file";
            return displayPreview(diskUsagePreview(target.toString(), type, stats));
        } catch (Exception e) {
            return new DisplayData("Unable to compute disk usage: " + e.getMessage());
        }
    }

    private DisplayData renderCat(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %cat <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }

            TextPreview preview = readUtf8Preview(target, MAX_CAT_BYTES);
            return displayPreview(filePreview(target.toString(), preview));
        } catch (Exception e) {
            return new DisplayData("Unable to read file: " + e.getMessage());
        }
    }

    private DisplayData renderHead(String raw) {
        LinePreviewOptions options;
        try {
            options = parseLinePreviewOptions("head", raw);
        } catch (IllegalArgumentException e) {
            return new DisplayData(e.getMessage());
        }
        try {
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            List<String> lines = Files.readAllLines(target, java.nio.charset.StandardCharsets.UTF_8);
            List<String> preview = lines.stream().limit(options.lines()).toList();
            return displayPreview(linePreview("Head", target.toString(), preview, lines.size()));
        } catch (Exception e) {
            return new DisplayData("Unable to read file head: " + e.getMessage());
        }
    }

    private DisplayData renderTail(String raw) {
        LinePreviewOptions options;
        try {
            options = parseLinePreviewOptions("tail", raw);
        } catch (IllegalArgumentException e) {
            return new DisplayData(e.getMessage());
        }
        try {
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            List<String> lines = Files.readAllLines(target, java.nio.charset.StandardCharsets.UTF_8);
            int fromIndex = Math.max(0, lines.size() - options.lines());
            List<String> preview = lines.subList(fromIndex, lines.size());
            return displayPreview(linePreview("Tail", target.toString(), preview, lines.size()));
        } catch (Exception e) {
            return new DisplayData("Unable to read file tail: " + e.getMessage());
        }
    }

    private DisplayData renderGrep(String raw) {
        SearchOptions options;
        try {
            options = parseSearchOptions("grep", raw);
        } catch (IllegalArgumentException e) {
            return new DisplayData(e.getMessage());
        }
        try {
            Path target = options.path() == null
                    ? currentDirectory
                    : resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("Grep target not found: " + target);
            }

            List<String> matches = new java.util.ArrayList<>();
            if (Files.isRegularFile(target)) {
                collectGrepMatches(target, target.getFileName().toString(), options.pattern(), matches, 50);
            } else if (Files.isDirectory(target)) {
                try (Stream<Path> stream = Files.walk(target, 3)) {
                    stream.filter(Files::isRegularFile)
                            .sorted(Comparator.comparing(Path::toString, String.CASE_INSENSITIVE_ORDER))
                            .limit(80)
                            .forEach(path -> collectGrepMatches(path, target.relativize(path).toString(), options.pattern(), matches, 50));
                }
            } else {
                return new DisplayData("Grep target is not a file or directory: " + target);
            }

            return displayPreview(matchesPreview("Grep", "Grep", options.pattern(), matches));
        } catch (Exception e) {
            return new DisplayData("Unable to search text: " + e.getMessage());
        }
    }

    private DisplayData renderFindFile(String raw) {
        SearchOptions options;
        try {
            options = parseSearchOptions("findfile", raw);
        } catch (IllegalArgumentException e) {
            return new DisplayData(e.getMessage());
        }
        try {
            Path target = options.path() == null
                    ? currentDirectory
                    : resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("Find target not found: " + target);
            }

            List<String> matches = new java.util.ArrayList<>();
            if (Files.isRegularFile(target)) {
                String name = target.getFileName().toString();
                if (name.contains(options.pattern())) {
                    matches.add(name);
                }
            } else if (Files.isDirectory(target)) {
                try (Stream<Path> stream = Files.walk(target, 3)) {
                    stream.filter(path -> !path.equals(target))
                            .sorted(Comparator.comparing(Path::toString, String.CASE_INSENSITIVE_ORDER))
                            .limit(120)
                            .forEach(path -> {
                                String relative = target.relativize(path).toString();
                                if (relative.contains(options.pattern())) {
                                    matches.add(Files.isDirectory(path) ? relative + "/" : relative);
                                }
                            });
                }
            } else {
                return new DisplayData("Find target is not a file or directory: " + target);
            }

            return displayPreview(matchesPreview("FindFile", "Find File", options.pattern(), matches));
        } catch (Exception e) {
            return new DisplayData("Unable to search file names: " + e.getMessage());
        }
    }

    private DisplayData renderWordCount(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %wc <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }

            String text = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            long lines = text.isEmpty() ? 0 : text.lines().count();
            long words = text.isBlank() ? 0 : java.util.Arrays.stream(text.trim().split("\\s+"))
                    .filter(token -> !token.isBlank())
                    .count();
            int chars = text.length();
            long bytes = Files.size(target);

            return displayPreview(wordCountPreview(target.toString(), lines, words, chars, bytes));
        } catch (Exception e) {
            return new DisplayData("Unable to compute file stats: " + e.getMessage());
        }
    }

    private DisplayData renderSha256(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %sha256 <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }

            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
            try (java.io.InputStream in = Files.newInputStream(target)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = in.read(buffer)) >= 0) {
                    if (read > 0) {
                        digest.update(buffer, 0, read);
                    }
                }
            }
            String hash = toHex(digest.digest());
            long bytes = Files.size(target);
            return displayPreview(sha256Preview(target.toString(), bytes, hash));
        } catch (Exception e) {
            return new DisplayData("Unable to compute SHA-256: " + e.getMessage());
        }
    }

    private String toHex(byte[] bytes) {
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            out.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return out.toString();
    }

    private DisplayData renderDiff(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %diff <LEFT_PATH> <RIGHT_PATH>");
        }
        int split = raw.indexOf(' ');
        if (split <= 0) {
            return new DisplayData("Usage: %diff <LEFT_PATH> <RIGHT_PATH>");
        }
        String leftPart = raw.substring(0, split).trim();
        String rightPart = raw.substring(split + 1).trim();
        if (leftPart.isBlank() || rightPart.isBlank()) {
            return new DisplayData("Usage: %diff <LEFT_PATH> <RIGHT_PATH>");
        }
        try {
            Path left = resolveNotebookPath(leftPart).toAbsolutePath().normalize();
            Path right = resolveNotebookPath(rightPart).toAbsolutePath().normalize();
            if (!Files.exists(left)) {
                return new DisplayData("Left file not found: " + left);
            }
            if (!Files.exists(right)) {
                return new DisplayData("Right file not found: " + right);
            }
            if (!Files.isRegularFile(left)) {
                return new DisplayData("Left path is not a file: " + left);
            }
            if (!Files.isRegularFile(right)) {
                return new DisplayData("Right path is not a file: " + right);
            }

            List<String> leftLines = Files.readAllLines(left, java.nio.charset.StandardCharsets.UTF_8);
            List<String> rightLines = Files.readAllLines(right, java.nio.charset.StandardCharsets.UTF_8);
            int max = Math.max(leftLines.size(), rightLines.size());
            List<String> diffLines = new java.util.ArrayList<>();
            int changed = 0;
            int added = 0;
            int removed = 0;
            for (int i = 0; i < max; i++) {
                String leftLine = i < leftLines.size() ? leftLines.get(i) : null;
                String rightLine = i < rightLines.size() ? rightLines.get(i) : null;
                if (java.util.Objects.equals(leftLine, rightLine)) {
                    continue;
                }
                changed++;
                if (leftLine != null) {
                    diffLines.add("- " + (i + 1) + ": " + leftLine);
                    removed++;
                }
                if (rightLine != null) {
                    diffLines.add("+ " + (i + 1) + ": " + rightLine);
                    added++;
                }
            }

            return displayPreview(diffPreview(
                    left.toString(),
                    right.toString(),
                    changed,
                    added,
                    removed,
                    diffLines
            ));
        } catch (Exception e) {
            return new DisplayData("Unable to diff files: " + e.getMessage());
        }
    }

    private DisplayData renderZipList(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %zipls <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }

            ArchiveListing listing = listZip(target);
            NotebookPreview preview = archiveListingPreview("Zip", target.toString(), listing);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to inspect zip file: " + e.getMessage());
        }
    }

    private DisplayData renderZipCat(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %zipcat <ZIP_PATH> <ENTRY_PATH>");
        }
        int split = raw.indexOf(' ');
        if (split <= 0) {
            return new DisplayData("Usage: %zipcat <ZIP_PATH> <ENTRY_PATH>");
        }
        String zipPart = raw.substring(0, split).trim();
        String entryPart = raw.substring(split + 1).trim();
        if (zipPart.isBlank() || entryPart.isBlank()) {
            return new DisplayData("Usage: %zipcat <ZIP_PATH> <ENTRY_PATH>");
        }
        try {
            Path target = resolveNotebookPath(zipPart).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("Zip file not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }

            ArchiveEntry entry = readZipEntry(target, entryPart);
            if (entry == null) {
                return new DisplayData("Zip entry not found: " + entryPart);
            }
            if (entry.directory()) {
                return new DisplayData("Zip entry is a directory: " + entryPart);
            }
            byte[] bytes = entry.bytes();
            boolean truncated = bytes.length > MAX_CAT_BYTES;
            byte[] previewBytes = truncated ? java.util.Arrays.copyOf(bytes, MAX_CAT_BYTES) : bytes;
            NotebookPreview preview = archiveEntryPreview(
                    "ZipEntry",
                    "Zip Entry",
                    target.toString(),
                    entryPart,
                    bytes.length,
                    previewBytes,
                    truncated
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to read zip entry: " + e.getMessage());
        }
    }

    private DisplayData renderTarList(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %tarls <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }

            ArchiveListing listing = listTar(target);
            NotebookPreview preview = archiveListingPreview("Tar", target.toString(), listing);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to inspect tar file: " + e.getMessage());
        }
    }

    private DisplayData renderTarCat(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %tarcat <TAR_PATH> <ENTRY_PATH>");
        }
        int split = raw.indexOf(' ');
        if (split <= 0) {
            return new DisplayData("Usage: %tarcat <TAR_PATH> <ENTRY_PATH>");
        }
        String tarPart = raw.substring(0, split).trim();
        String entryPart = raw.substring(split + 1).trim();
        if (tarPart.isBlank() || entryPart.isBlank()) {
            return new DisplayData("Usage: %tarcat <TAR_PATH> <ENTRY_PATH>");
        }
        try {
            Path target = resolveNotebookPath(tarPart).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("Tar file not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }

            ArchiveEntry entry = readTarEntryPreview(target, entryPart, MAX_CAT_BYTES);
            if (entry == null) {
                return new DisplayData("Tar entry not found: " + entryPart);
            }
            if (entry.directory()) {
                return new DisplayData("Tar entry is a directory: " + entryPart);
            }
            boolean truncated = entry.size() > MAX_CAT_BYTES;
            NotebookPreview preview = archiveEntryPreview(
                    "TarEntry",
                    "Tar Entry",
                    target.toString(),
                    entryPart,
                    entry.size(),
                    entry.bytes(),
                    truncated
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to read tar entry: " + e.getMessage());
        }
    }

    private DisplayData renderExtract(String raw) {
        ExtractOptions options;
        try {
            options = parseExtractOptions(raw);
        } catch (IllegalArgumentException e) {
            return new DisplayData(e.getMessage());
        }
        try {
            Path archive = resolveNotebookPath(options.archivePath()).toAbsolutePath().normalize();
            if (!Files.exists(archive)) {
                return new DisplayData("Archive not found: " + archive);
            }
            if (!Files.isRegularFile(archive)) {
                return new DisplayData("Archive path is not a file: " + archive);
            }
            Path output = resolveExtractOutputPath(options.entryPath(), options.outputPath());
            ArchiveEntry entry = readArchiveEntryBytes(archive, options.entryPath(), MAX_EXTRACT_BYTES);
            if (entry == null) {
                return new DisplayData("Archive entry not found: " + options.entryPath());
            }
            if (entry.directory()) {
                return new DisplayData("Archive entry is a directory: " + options.entryPath());
            }
            if (entry.bytes().length > MAX_EXTRACT_BYTES) {
                return new DisplayData("Archive entry is too large to extract in notebook: " + entry.bytes().length + " bytes");
            }

            if (!options.dryRun()) {
                if (Files.exists(output)) {
                    return new DisplayData("Output already exists: " + output);
                }
                Path parent = output.getParent();
                if (parent != null) {
                    Files.createDirectories(parent);
                }
                Files.write(output, entry.bytes());
            }

            NotebookPreview preview = extractPreview(
                    options.dryRun(),
                    archive.toString(),
                    options.entryPath(),
                    output.toString(),
                    entry.bytes().length
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to extract archive entry: " + e.getMessage());
        }
    }

    private Path resolveExtractOutputPath(String entryPart, String outputPart) {
        String rawOutput = outputPart == null || outputPart.isBlank() ? entryPart : outputPart;
        Path output = resolveNotebookPath(rawOutput).toAbsolutePath().normalize();
        Path root = currentDirectory.toAbsolutePath().normalize();
        if (!output.startsWith(root)) {
            throw new IllegalArgumentException("Extraction output must stay under notebook directory: " + root);
        }
        return output;
    }

    private DisplayData renderJson(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %json <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String text = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8).trim();
            if (text.isEmpty()) {
                return new DisplayData("JSON file is empty: " + target);
            }
            String pretty = prettyPrintJson(text);
            return displayPreview(jsonPreview(target.toString(), pretty));
        } catch (IllegalArgumentException e) {
            return new DisplayData("Unable to pretty-print JSON: " + e.getMessage());
        } catch (Exception e) {
            return new DisplayData("Unable to read JSON file: " + e.getMessage());
        }
    }

    private DisplayData renderCsv(String rawPath) {
        return renderDelimitedTable(rawPath, ",", "CSV");
    }

    private DisplayData renderDelimitedTable(String rawPath, String delimiter, String label) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %" + label.toLowerCase(Locale.ROOT) + " [-n N] [--profile] <PATH>");
        }
        try {
            TableOptions options = parseTableOptions(rawPath, label);
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, delimiter, label);
            if (table.isEmpty()) {
                return new DisplayData(label + " file is empty: " + target);
            }
            List<String> header = table.header();
            List<List<String>> allRows = table.rows();
            List<List<String>> previewRows = allRows.stream().limit(options.previewRows()).toList();
            List<ColumnProfile> profiles = options.profile() ? profileColumns(header, allRows) : List.of();
            NotebookPreview preview = delimitedTablePreview(
                    label,
                    target.toString(),
                    header,
                    allRows,
                    previewRows,
                    options.profile(),
                    profiles
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview " + label + ": " + e.getMessage());
        }
    }

    private DisplayData renderLinePlot(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>");
        }
        try {
            boolean tsv = false;
            String remaining = raw.trim();
            if (remaining.startsWith("--tsv ")) {
                tsv = true;
                remaining = remaining.substring("--tsv".length()).trim();
            }
            String[] parts = remaining.split("\\s+");
            if (parts.length != 3) {
                return new DisplayData("Usage: %lineplot [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>");
            }
            Path target = resolveNotebookPath(parts[0]).toAbsolutePath().normalize();
            String xColumn = parts[1];
            String yColumn = parts[2];
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, tsv);
            if (table.isEmpty()) {
                return new DisplayData("Line plot source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            int xIndex = header.indexOf(xColumn);
            int yIndex = header.indexOf(yColumn);
            if (xIndex < 0 || yIndex < 0) {
                return new DisplayData("Column not found. Requested x=" + xColumn + " y=" + yColumn
                        + "\nAvailable columns: " + String.join(", ", header));
            }
            List<LinePlotPoint> points = new java.util.ArrayList<>();
            int skipped = 0;
            for (List<String> row : table.rows()) {
                String yValue = getCell(row, yIndex).trim();
                if (yValue.isEmpty()) {
                    skipped++;
                    continue;
                }
                try {
                    double parsed = Double.parseDouble(yValue);
                    if (Double.isFinite(parsed)) {
                        points.add(new LinePlotPoint(getCell(row, xIndex), parsed));
                    } else {
                        skipped++;
                    }
                } catch (NumberFormatException e) {
                    skipped++;
                }
            }
            if (points.isEmpty()) {
                return new DisplayData("No numeric values found for " + yColumn + " in " + target);
            }

            NotebookPreview preview = linePlotPreview(target.toString(), label, xColumn, yColumn, points, skipped);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to render line plot: " + e.getMessage());
        }
    }

    private DisplayData renderSample(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %sample [--tsv] [-n N] [--seed S] <PATH>");
        }
        try {
            SampleOptions options = parseSampleOptions(raw);
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, options.tsv());
            if (table.isEmpty()) {
                return new DisplayData("Sample source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            List<List<String>> allRows = table.rows();
            if (allRows.isEmpty()) {
                return new DisplayData("No data rows found in " + target);
            }
            List<List<String>> sampledRows = sampleRows(allRows, options.rows(), options.seed());
            String seedLabel = options.seed() == null ? "random" : Long.toString(options.seed());
            NotebookPreview preview = samplePreview(
                    target.toString(),
                    label,
                    header,
                    allRows.size(),
                    sampledRows,
                    seedLabel
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to sample rows: " + e.getMessage());
        }
    }

    private DisplayData renderSort(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %sort [--tsv] [-n N] [--desc] <PATH> <COLUMN>");
        }
        try {
            SortOptions options = parseSortOptions(raw);
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, options.tsv());
            if (table.isEmpty()) {
                return new DisplayData("Sort source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            int columnIndex = header.indexOf(options.column());
            if (columnIndex < 0) {
                return new DisplayData("Column not found: " + options.column()
                        + "\nAvailable columns: " + String.join(", ", header));
            }
            List<List<String>> rows = table.rows();
            if (rows.isEmpty()) {
                return new DisplayData("No data rows found in " + target);
            }

            boolean numeric = isNumericSortColumn(rows, columnIndex);
            List<SortRow> sortedRows = new java.util.ArrayList<>();
            for (int i = 0; i < rows.size(); i++) {
                List<String> row = rows.get(i);
                String key = getCell(row, columnIndex).trim();
                sortedRows.add(new SortRow(row, key, numeric ? parseFiniteDouble(key) : null, i));
            }
            sortedRows.sort((left, right) -> compareSortRows(left, right, numeric, options.descending()));
            List<SortRow> visibleRows = sortedRows.stream().limit(options.rows()).toList();
            String order = options.descending() ? "desc" : "asc";
            String mode = numeric ? "numeric" : "text";
            List<List<String>> visibleCells = visibleRows.stream().map(SortRow::cells).toList();
            NotebookPreview preview = sortPreview(
                    target.toString(),
                    label,
                    options.column(),
                    order,
                    mode,
                    header,
                    rows.size(),
                    visibleCells
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to sort rows: " + e.getMessage());
        }
    }

    private DisplayData renderFilter(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %filter [--tsv] [-n N] <PATH> <COLUMN> <OP> [VALUE]");
        }
        try {
            FilterOptions options = parseFilterOptions(raw);
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, options.tsv());
            if (table.isEmpty()) {
                return new DisplayData("Filter source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            int columnIndex = header.indexOf(options.column());
            if (columnIndex < 0) {
                return new DisplayData("Column not found: " + options.column()
                        + "\nAvailable columns: " + String.join(", ", header));
            }
            List<List<String>> rows = new java.util.ArrayList<>();
            List<List<String>> matchedRows = new java.util.ArrayList<>();
            for (List<String> row : table.rows()) {
                rows.add(row);
                if (matchesFilter(getCell(row, columnIndex), options.operator(), options.value())) {
                    matchedRows.add(row);
                }
            }
            List<List<String>> visibleRows = matchedRows.stream().limit(options.rows()).toList();
            String predicate = filterPredicateLabel(options.column(), options.operator(), options.value());
            NotebookPreview preview = filterPreview(
                    target.toString(),
                    label,
                    options.column(),
                    options.operator(),
                    options.value(),
                    predicate,
                    header,
                    rows.size(),
                    matchedRows.size(),
                    visibleRows
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to filter rows: " + e.getMessage());
        }
    }

    private DisplayData renderSchema(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %schema [--tsv] <PATH>");
        }
        try {
            boolean tsv = false;
            String pathPart = raw.trim();
            if (pathPart.equals("--tsv") || pathPart.startsWith("--tsv ")) {
                tsv = true;
                pathPart = pathPart.substring("--tsv".length()).trim();
            }
            if (pathPart.isBlank()) {
                return new DisplayData("Usage: %schema [--tsv] <PATH>");
            }
            if (pathPart.startsWith("--")) {
                int split = pathPart.indexOf(' ');
                String option = split < 0 ? pathPart : pathPart.substring(0, split);
                return new DisplayData("Unknown %schema option: " + option + "\nUsage: %schema [--tsv] <PATH>");
            }
            Path target = resolveNotebookPath(pathPart).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, tsv);
            if (table.isEmpty()) {
                return new DisplayData("Schema source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            List<List<String>> rows = table.rows();
            if (header.isEmpty()) {
                return new DisplayData("Schema source has no columns: " + target);
            }
            List<SchemaColumn> columns = inferSchemaColumns(header, rows);
            NotebookPreview preview = schemaPreview(target.toString(), label, rows.size(), columns);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to infer schema: " + e.getMessage());
        }
    }

    private DisplayData renderMissing(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %missing [--tsv] <PATH>");
        }
        try {
            boolean tsv = false;
            String pathPart = raw.trim();
            if (pathPart.equals("--tsv") || pathPart.startsWith("--tsv ")) {
                tsv = true;
                pathPart = pathPart.substring("--tsv".length()).trim();
            }
            if (pathPart.isBlank()) {
                return new DisplayData("Usage: %missing [--tsv] <PATH>");
            }
            if (pathPart.startsWith("--")) {
                int split = pathPart.indexOf(' ');
                String option = split < 0 ? pathPart : pathPart.substring(0, split);
                return new DisplayData("Unknown %missing option: " + option + "\nUsage: %missing [--tsv] <PATH>");
            }
            Path target = resolveNotebookPath(pathPart).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, tsv);
            if (table.isEmpty()) {
                return new DisplayData("Missing summary source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            List<List<String>> rows = table.rows();
            if (header.isEmpty()) {
                return new DisplayData("Missing summary source has no columns: " + target);
            }
            List<MissingColumn> columns = new java.util.ArrayList<>();
            for (int column = 0; column < header.size(); column++) {
                int missing = 0;
                for (List<String> row : rows) {
                    if (getCell(row, column).trim().isEmpty()) {
                        missing++;
                    }
                }
                columns.add(new MissingColumn(header.get(column), missing, rows.size() - missing));
            }
            long columnsWithMissing = columns.stream().filter(column -> column.missing() > 0).count();
            NotebookPreview preview = missingPreview(
                    target.toString(),
                    label,
                    rows.size(),
                    columns,
                    columnsWithMissing,
                    buildMissingSvg(columns, rows.size(), MAX_MISSING_CHART_COLUMNS)
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to compute missing summary: " + e.getMessage());
        }
    }

    private DisplayData renderValueCounts(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %valuecounts [--tsv] [--top N] <PATH> <COLUMN>");
        }
        try {
            ValueCountsOptions options = parseValueCountsOptions(raw);
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, options.tsv());
            if (table.isEmpty()) {
                return new DisplayData("Value counts source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            int columnIndex = header.indexOf(options.column());
            if (columnIndex < 0) {
                return new DisplayData("Column not found: " + options.column()
                        + "\nAvailable columns: " + String.join(", ", header));
            }
            List<List<String>> rows = table.rows();
            if (rows.isEmpty()) {
                return new DisplayData("No data rows found for " + options.column() + " in " + target);
            }
            java.util.Map<String, Integer> counts = new java.util.LinkedHashMap<>();
            for (List<String> row : rows) {
                String value = getCell(row, columnIndex).trim();
                String key = value.isEmpty() ? "(blank)" : value;
                counts.merge(key, 1, Integer::sum);
            }
            List<ValueCount> sorted = counts.entrySet().stream()
                    .map(entry -> new ValueCount(entry.getKey(), entry.getValue()))
                    .sorted(Comparator.comparingInt(ValueCount::count).reversed().thenComparing(ValueCount::value))
                    .toList();
            List<ValueCount> visible = sorted.stream().limit(options.top()).toList();
            int visibleRows = visible.stream().mapToInt(ValueCount::count).sum();
            int otherRows = rows.size() - visibleRows;
            NotebookPreview preview = valueCountsPreview(
                    target.toString(),
                    label,
                    options.column(),
                    rows.size(),
                    sorted.size(),
                    visible,
                    otherRows,
                    buildValueCountsSvg(options.column(), visible, rows.size())
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to compute value counts: " + e.getMessage());
        }
    }

    private DisplayData renderGroupBy(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %groupby [--tsv] <PATH> <GROUP_COLUMN> [VALUE_COLUMN] [count|sum|mean|min|max]");
        }
        try {
            GroupByOptions options = parseGroupByOptions(raw);
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, options.tsv());
            if (table.isEmpty()) {
                return new DisplayData("GroupBy source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            int groupIndex = header.indexOf(options.groupColumn());
            if (groupIndex < 0) {
                return new DisplayData("Group column not found: " + options.groupColumn()
                        + "\nAvailable columns: " + String.join(", ", header));
            }
            int valueIndex = -1;
            if (options.valueColumn() != null) {
                valueIndex = header.indexOf(options.valueColumn());
                if (valueIndex < 0) {
                    return new DisplayData("Value column not found: " + options.valueColumn()
                            + "\nAvailable columns: " + String.join(", ", header));
                }
            }
            java.util.Map<String, GroupAccumulator> groups = new java.util.LinkedHashMap<>();
            int rows = 0;
            int skipped = 0;
            for (List<String> row : table.rows()) {
                rows++;
                String group = getCell(row, groupIndex).trim();
                String groupKey = group.isEmpty() ? "(blank)" : group;
                Double value = null;
                if (options.valueColumn() != null) {
                    value = parseFiniteDouble(getCell(row, valueIndex));
                    if (value == null) {
                        skipped++;
                    }
                }
                groups.computeIfAbsent(groupKey, ignored -> new GroupAccumulator()).add(value);
            }
            if (groups.isEmpty()) {
                return new DisplayData("No data rows found for " + options.groupColumn() + " in " + target);
            }
            List<GroupResult> sorted = groups.entrySet().stream()
                    .map(entry -> entry.getValue().toResult(entry.getKey()))
                    .sorted(groupComparator(options.aggregate()))
                    .toList();
            if (!options.aggregate().equals("count") && sorted.stream().allMatch(result -> result.numeric() == 0)) {
                return new DisplayData("No numeric values found for " + options.valueColumn() + " in " + target);
            }
            List<GroupResult> visible = sorted.stream().limit(MAX_GROUPBY_GROUPS).toList();
            int hiddenGroups = Math.max(0, sorted.size() - visible.size());
            NotebookPreview preview = groupByPreview(
                    target.toString(),
                    label,
                    options.groupColumn(),
                    options.valueColumn(),
                    options.aggregate(),
                    rows,
                    sorted.size(),
                    hiddenGroups,
                    skipped,
                    visible,
                    buildGroupBySvg(options.groupColumn(), options.aggregate(), visible)
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to compute groupby: " + e.getMessage());
        }
    }

    private DisplayData renderScatterPlot(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>");
        }
        try {
            boolean tsv = false;
            String remaining = raw.trim();
            if (remaining.startsWith("--tsv ")) {
                tsv = true;
                remaining = remaining.substring("--tsv".length()).trim();
            }
            String[] parts = remaining.split("\\s+");
            if (parts.length != 3) {
                return new DisplayData("Usage: %scatter [--tsv] <PATH> <X_COLUMN> <Y_COLUMN>");
            }
            Path target = resolveNotebookPath(parts[0]).toAbsolutePath().normalize();
            String xColumn = parts[1];
            String yColumn = parts[2];
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, tsv);
            if (table.isEmpty()) {
                return new DisplayData("Scatter plot source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            int xIndex = header.indexOf(xColumn);
            int yIndex = header.indexOf(yColumn);
            if (xIndex < 0 || yIndex < 0) {
                return new DisplayData("Column not found. Requested x=" + xColumn + " y=" + yColumn
                        + "\nAvailable columns: " + String.join(", ", header));
            }
            List<ScatterPoint> points = new java.util.ArrayList<>();
            int skipped = 0;
            for (List<String> row : table.rows()) {
                Double x = parseFiniteDouble(getCell(row, xIndex));
                Double y = parseFiniteDouble(getCell(row, yIndex));
                if (x == null || y == null) {
                    skipped++;
                    continue;
                }
                points.add(new ScatterPoint(x, y));
            }
            if (points.isEmpty()) {
                return new DisplayData("No paired numeric values found for " + xColumn + " and " + yColumn + " in " + target);
            }

            NotebookPreview preview = scatterPlotPreview(target.toString(), label, xColumn, yColumn, points, skipped);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to render scatter plot: " + e.getMessage());
        }
    }

    private DisplayData renderHistogram(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %hist [--tsv] [--bins N] <PATH> <COLUMN>");
        }
        try {
            HistogramOptions options = parseHistogramOptions(raw);
            Path target = resolveNotebookPath(options.path()).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, options.tsv());
            if (table.isEmpty()) {
                return new DisplayData("Histogram source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            int columnIndex = header.indexOf(options.column());
            if (columnIndex < 0) {
                return new DisplayData("Column not found: " + options.column()
                        + "\nAvailable columns: " + String.join(", ", header));
            }
            List<Double> values = new java.util.ArrayList<>();
            int skipped = 0;
            for (List<String> row : table.rows()) {
                String value = getCell(row, columnIndex).trim();
                if (value.isEmpty()) {
                    skipped++;
                    continue;
                }
                try {
                    double parsed = Double.parseDouble(value);
                    if (Double.isFinite(parsed)) {
                        values.add(parsed);
                    } else {
                        skipped++;
                    }
                } catch (NumberFormatException e) {
                    skipped++;
                }
            }
            if (values.isEmpty()) {
                return new DisplayData("No numeric values found for " + options.column() + " in " + target);
            }
            double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
            double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            List<HistogramBin> bins = buildHistogramBins(values, options.bins(), min, max);
            NotebookPreview preview = histogramPreview(
                    target.toString(),
                    label,
                    options.column(),
                    values.size(),
                    skipped,
                    bins,
                    min,
                    max
            );
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to render histogram: " + e.getMessage());
        }
    }

    private DisplayData renderDescribe(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %describe [--tsv] <PATH>");
        }
        try {
            boolean tsv = false;
            String pathPart = raw.trim();
            if (pathPart.startsWith("--tsv ")) {
                tsv = true;
                pathPart = pathPart.substring("--tsv".length()).trim();
            }
            if (pathPart.isBlank()) {
                return new DisplayData("Usage: %describe [--tsv] <PATH>");
            }
            Path target = resolveNotebookPath(pathPart).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, tsv);
            if (table.isEmpty()) {
                return new DisplayData("Describe source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            List<List<String>> rows = table.rows();
            List<NumericSummary> summaries = describeNumericColumns(header, rows);
            if (summaries.isEmpty()) {
                return new DisplayData("No numeric columns found in " + target);
            }
            NotebookPreview preview = describePreview(target.toString(), label, rows.size(), header.size(), summaries);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to describe table: " + e.getMessage());
        }
    }

    private DisplayData renderCorrelation(String raw) {
        if (raw == null || raw.isBlank()) {
            return new DisplayData("Usage: %corr [--tsv] [--heatmap] <PATH>");
        }
        try {
            boolean tsv = false;
            boolean heatmap = false;
            String pathPart = raw.trim();
            while (pathPart.startsWith("--")) {
                if (pathPart.equals("--tsv") || pathPart.startsWith("--tsv ")) {
                    tsv = true;
                    pathPart = pathPart.substring("--tsv".length()).trim();
                    continue;
                }
                if (pathPart.equals("--heatmap") || pathPart.startsWith("--heatmap ")) {
                    heatmap = true;
                    pathPart = pathPart.substring("--heatmap".length()).trim();
                    continue;
                }
                int split = pathPart.indexOf(' ');
                String option = split < 0 ? pathPart : pathPart.substring(0, split);
                return new DisplayData("Unknown %corr option: " + option + "\nUsage: %corr [--tsv] [--heatmap] <PATH>");
            }
            if (pathPart.isBlank()) {
                return new DisplayData("Usage: %corr [--tsv] [--heatmap] <PATH>");
            }
            Path target = resolveNotebookPath(pathPart).toAbsolutePath().normalize();
            String fileError = regularFileError(target);
            if (fileError != null) {
                return new DisplayData(fileError);
            }
            DelimitedTable table = readDelimitedTable(target, tsv);
            if (table.isEmpty()) {
                return new DisplayData("Correlation source is empty: " + target);
            }
            String label = table.label();
            List<String> header = table.header();
            List<List<String>> rows = table.rows();
            List<NumericColumn> numericColumns = findNumericColumns(header, rows);
            if (numericColumns.size() < 2) {
                return new DisplayData("Need at least two numeric columns for correlation in " + target);
            }
            Double[][] correlations = new Double[numericColumns.size()][numericColumns.size()];
            for (int i = 0; i < numericColumns.size(); i++) {
                for (int j = 0; j < numericColumns.size(); j++) {
                    correlations[i][j] = correlation(rows, numericColumns.get(i).index(), numericColumns.get(j).index());
                }
            }

            NotebookPreview preview = correlationPreview(target.toString(), label, rows.size(), numericColumns, correlations, heatmap);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to compute correlation: " + e.getMessage());
        }
    }

    private DisplayData renderMarkdown(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %md <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String markdown = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = markdownPreview(target.toString(), markdown);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview Markdown: " + e.getMessage());
        }
    }

    private DisplayData renderHtmlFile(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %html <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String htmlSource = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = htmlPreview(target.toString(), htmlSource);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview HTML: " + e.getMessage());
        }
    }

    private DisplayData renderYamlFile(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %yaml <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String yaml = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = yamlPreview(target.toString(), yaml);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview YAML: " + e.getMessage());
        }
    }

    private DisplayData renderTomlFile(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %toml <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String toml = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = tomlPreview(target.toString(), toml);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview TOML: " + e.getMessage());
        }
    }

    private DisplayData renderXmlFile(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %xml <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String xml = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = xmlPreview(target.toString(), xml);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview XML: " + e.getMessage());
        }
    }

    private DisplayData renderIniFile(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %ini <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String ini = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = iniPreview(target.toString(), ini);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview INI: " + e.getMessage());
        }
    }

    private DisplayData renderPropertiesFile(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %properties <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String source = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = propertiesPreview(target.toString(), source);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview properties: " + e.getMessage());
        }
    }

    private DisplayData renderEnvFile(String rawPath) {
        if (rawPath == null || rawPath.isBlank()) {
            return new DisplayData("Usage: %envfile <PATH>");
        }
        try {
            Path target = resolveNotebookPath(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(target)) {
                return new DisplayData("File not found: " + target);
            }
            if (!Files.isRegularFile(target)) {
                return new DisplayData("Not a file: " + target);
            }
            String source = Files.readString(target, java.nio.charset.StandardCharsets.UTF_8);
            NotebookPreview preview = envFilePreview(target.toString(), source);
            return displayPreview(preview);
        } catch (Exception e) {
            return new DisplayData("Unable to preview env file: " + e.getMessage());
        }
    }

    private Path resolveNotebookPath(String rawPath) {
        return Path.of(rawPath).isAbsolute()
                ? Path.of(rawPath).normalize()
                : currentDirectory.resolve(rawPath).normalize();
    }

    private DisplayData renderEnv(String name) {
        if (name == null || name.isBlank()) {
            return new DisplayData("Usage: %env <NAME>");
        }
        String value = System.getenv(name);
        return displayPreview(envPreview(name, value));
    }

    private DisplayData renderWhichClass(String className) {
        if (className == null || className.isBlank()) {
            return new DisplayData("Usage: %whichclass <fully.qualified.ClassName>");
        }
        try {
            Object locationResult = evaluator.eval(className + ".class.getProtectionDomain().getCodeSource() == null"
                    + " ? \"<unknown>\""
                    + " : " + className + ".class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath()");
            String location = locationResult == null ? "<unknown>" : locationResult.toString();
            return displayPreview(classLocationPreview(className, location));
        } catch (Throwable e) {
            return new DisplayData("Class not found: " + className);
        }
    }

    private static String escapeHtml(String value) {
        return NotebookHtml.escapeText(value);
    }
}
