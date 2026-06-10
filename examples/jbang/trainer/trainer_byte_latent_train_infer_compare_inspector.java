//DEPS com.fasterxml.jackson.core:jackson-databind:2.17.2

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.attribute.FileAttribute;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

/* JADX INFO: loaded from: trainer_byte_latent_train_infer_compare_inspector.jar:trainer_byte_latent_train_infer_compare_inspector.class */
public class trainer_byte_latent_train_infer_compare_inspector {
    private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private static final ObjectMapper JSON_LINE = new ObjectMapper();
    private static final String REPORT_FILE_NAME = "train-infer-report.json";
    private static final String BUNDLE_MANIFEST_FILE_NAME = "manifest.json";
    private static final String COMPARE_BUNDLE_REPORT_VERSION = "train-infer-compare-bundle/v1";
    private static final String COMPARE_RESULT_REPORT_VERSION = "train-infer-compare-result/v1";
    private static final String SINGLE_REPORT_VERSION_FALLBACK = "train-infer-report/v1";

    public static void main(String[] args) {
        if (args.length == 0) {
            throw new IllegalArgumentException("Pass either a compare result path, a compare bundle path, or the left and right checkpoint directories or train-infer-report.json paths.");
        }
        if (isResultInput(Path.of(args[0], new String[0]))) {
            runResultMode(args);
        } else if (isBundleInput(Path.of(args[0], new String[0]))) {
            runBundleMode(args);
        } else {
            if (args.length < 2) {
                throw new IllegalArgumentException("Pass the left and right checkpoint directories or train-infer-report.json paths as the first two arguments.");
            }
            runCompareMode(args);
        }
    }

    private static void runCompareMode(String[] args) {
        CliOptions options = parseArgs(args, false);
        Path leftInputPath = Path.of(args[0], new String[0]);
        Path rightInputPath = Path.of(args[1], new String[0]);
        String section = options.section();
        String format = normalizeFormat(options.format(), options.outputPath());
        InputBundle left = resolveInput(leftInputPath);
        InputBundle right = resolveInput(rightInputPath);
        String normalizedSection = section == null ? "overview" : section.toLowerCase(Locale.ROOT);
        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Train+Infer Compare Inspector");
        System.out.println("====================================================");
        System.out.println("mode=compare");
        System.out.println("leftInputPath=" + String.valueOf(leftInputPath.toAbsolutePath().normalize()));
        System.out.println("rightInputPath=" + String.valueOf(rightInputPath.toAbsolutePath().normalize()));
        System.out.println("leftReportFile=" + String.valueOf(left.reportFile().toAbsolutePath().normalize()));
        System.out.println("rightReportFile=" + String.valueOf(right.reportFile().toAbsolutePath().normalize()));
        System.out.println("section=" + (section == null ? "overview" : section));
        System.out.println("format=" + format);
        if (normalizedSection.startsWith("bundle=")) {
            if (options.outputPath() == null) {
                throw new IllegalArgumentException("Bundle export requires an output directory path.");
            }
            Map<String, Object> manifest = writeBundle(left, right, normalizedSection, options.outputPath());
            System.out.println("wroteBundle=" + String.valueOf(options.outputPath().toAbsolutePath().normalize()));
            System.out.print(renderValue(manifest, format));
            return;
        }
        if (normalizedSection.startsWith("result=")) {
            if (options.outputPath() == null) {
                throw new IllegalArgumentException("Result export requires an output file path.");
            }
            Map<String, Object> result = writeResult(left, right, normalizedSection, options.outputPath());
            System.out.println("wroteResult=" + String.valueOf(options.outputPath().toAbsolutePath().normalize()));
            System.out.print(renderValue(result, format));
            return;
        }
        Object value = select(left, right, section);
        emitValue(value, format, options.outputPath());
    }

    private static void runBundleMode(String[] args) {
        CliOptions options = parseArgs(args, true);
        Path bundleInputPath = Path.of(args[0], new String[0]);
        String section = options.section();
        String format = normalizeFormat(options.format(), options.outputPath());
        BundleInput bundle = resolveBundleInput(bundleInputPath);
        Object value = selectBundle(bundle, section);
        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Train+Infer Compare Inspector");
        System.out.println("====================================================");
        System.out.println("mode=bundle");
        System.out.println("bundleInputPath=" + String.valueOf(bundleInputPath.toAbsolutePath().normalize()));
        System.out.println("bundleDir=" + String.valueOf(bundle.bundleDir().toAbsolutePath().normalize()));
        System.out.println("manifestFile=" + String.valueOf(bundle.manifestFile().toAbsolutePath().normalize()));
        System.out.println("section=" + (section == null ? "bundleSummary" : section));
        System.out.println("format=" + format);
        emitValue(value, format, options.outputPath());
    }

    private static void runResultMode(String[] args) {
        CliOptions options = parseArgs(args, true);
        Path resultInputPath = Path.of(args[0], new String[0]);
        String section = options.section();
        if (section == null || "bundlesummary".equalsIgnoreCase(section)) {
            section = "resultSummary";
        }
        String format = normalizeFormat(options.format(), options.outputPath());
        ResultInput result = resolveResultInput(resultInputPath);
        Object value = selectResult(result, section);
        System.out.println("====================================================");
        System.out.println(" Gollek Byte-Latent Train+Infer Compare Inspector");
        System.out.println("====================================================");
        System.out.println("mode=result");
        System.out.println("resultInputPath=" + String.valueOf(resultInputPath.toAbsolutePath().normalize()));
        System.out.println("resultFile=" + String.valueOf(result.resultFile().toAbsolutePath().normalize()));
        System.out.println("section=" + (section == null ? "resultSummary" : section));
        System.out.println("format=" + format);
        emitValue(value, format, options.outputPath());
    }

    private static InputBundle resolveInput(Path inputPath) {
        Path reportFile = inputPath;
        if (Files.isDirectory(inputPath, new LinkOption[0])) {
            reportFile = inputPath.resolve(REPORT_FILE_NAME);
        }
        if (!Files.isRegularFile(reportFile, new LinkOption[0])) {
            throw new IllegalArgumentException("Train+infer report not found: " + String.valueOf(reportFile.toAbsolutePath().normalize()));
        }
        return new InputBundle(inputPath, reportFile, loadJsonMap(reportFile));
    }

    private static boolean isBundleInput(Path inputPath) {
        if (Files.isDirectory(inputPath, new LinkOption[0])) {
            return Files.isRegularFile(inputPath.resolve(BUNDLE_MANIFEST_FILE_NAME), new LinkOption[0]);
        }
        return Files.isRegularFile(inputPath, new LinkOption[0]) && BUNDLE_MANIFEST_FILE_NAME.equalsIgnoreCase(String.valueOf(inputPath.getFileName()));
    }

    private static boolean isResultInput(Path inputPath) {
        if (!Files.isRegularFile(inputPath, new LinkOption[0])) {
            return false;
        }
        try {
            Map<String, Object> report = loadJsonMap(inputPath);
            return COMPARE_RESULT_REPORT_VERSION.equals(String.valueOf(report.getOrDefault("reportVersion", "")));
        } catch (RuntimeException e) {
            return false;
        }
    }

    private static BundleInput resolveBundleInput(Path inputPath) {
        Path manifestFile = inputPath;
        Path bundleDir = inputPath;
        if (Files.isDirectory(inputPath, new LinkOption[0])) {
            manifestFile = inputPath.resolve(BUNDLE_MANIFEST_FILE_NAME);
        } else {
            bundleDir = inputPath.toAbsolutePath().normalize().getParent();
        }
        if (bundleDir == null || !Files.isRegularFile(manifestFile, new LinkOption[0])) {
            throw new IllegalArgumentException("Compare bundle manifest not found: " + String.valueOf(manifestFile.toAbsolutePath().normalize()));
        }
        return new BundleInput(bundleDir, manifestFile, loadJsonMap(manifestFile));
    }

    private static ResultInput resolveResultInput(Path inputPath) {
        Path resultFile = inputPath.toAbsolutePath().normalize();
        if (!Files.isRegularFile(resultFile, new LinkOption[0])) {
            throw new IllegalArgumentException("Compare result not found: " + String.valueOf(resultFile));
        }
        return new ResultInput(resultFile, loadJsonMap(resultFile));
    }

    private static Object select(InputBundle left, InputBundle right, String section) {
        String normalized;
        normalized = section == null ? "overview" : section.toLowerCase(Locale.ROOT);
        switch (normalized) {
            case "overview":
                return summarizeOverview(left, right);
            case "comparison":
                return comparisonBlock(left, right);
            case "artifactsummary":
                return compareArtifactSummary(left, right);
            case "artifactsummaryline":
                return compareArtifactSummaryLine(left, right);
            case "comparesummary":
                return compareSummary(left, right);
            case "comparesummaryline":
                return compareSummaryLine(left, right);
            case "commands":
                return compareCommands(left, right, false);
            case "commands:ci":
                return compareCommands(left, right, true);
            case "selectors":
                return compareSelectors(left, right, false);
            case "selectors:ci":
                return compareSelectors(left, right, true);
            case "inspectfields":
                return compareInspectFields(false);
            case "inspectfields:ci":
                return compareInspectFields(true);
            case "status":
                return summarizeStatus(left, right, false);
            case "health":
                return summarizeStatus(left, right, false);
            case "ci":
            case "status:ci":
            case "health:ci":
                return summarizeStatus(left, right, true);
            case "runs":
                return List.of(summarizeRun("left", left), summarizeRun("right", right));
            case "left":
                return summarizeRun("left", left);
            case "right":
                return summarizeRun("right", right);
            case "delta:latesttrainloss":
                return summarizeNumericDelta("latestTrainLoss", asDouble(selectObjectPath(left.report().get("training"), "latestTrainLoss")), asDouble(selectObjectPath(right.report().get("training"), "latestTrainLoss")));
            case "delta:bestvalidationloss":
                return summarizeNumericDelta("bestValidationLoss", asDouble(selectObjectPath(left.report().get("training"), "bestValidationLoss")), asDouble(selectObjectPath(right.report().get("training"), "bestValidationLoss")));
            case "delta:epochcount":
                return summarizeNumericDelta("epochCount", asDouble(selectObjectPath(left.report().get("training"), "epochCount")), asDouble(selectObjectPath(right.report().get("training"), "epochCount")));
            case "delta:nexttoken":
                return summarizeValueDelta("nextToken", selectObjectPath(left.report().get("inference"), "nextToken"), selectObjectPath(right.report().get("inference"), "nextToken"));
            case "delta:nexttokens":
                return summarizeValueDelta("nextTokens", selectObjectPath(left.report().get("inference"), "nextTokens"), selectObjectPath(right.report().get("inference"), "nextTokens"));
            case "delta:nexttokenspreview":
                return summarizeValueDelta("nextTokensPreview", readNextTokensPreview(asMap(left.report().get("inference"))), readNextTokensPreview(asMap(right.report().get("inference"))));
            case "delta:nexttokenscount":
                return summarizeNumericDelta("nextTokensCount", sizeOfValue(selectObjectPath(left.report().get("inference"), "nextTokens")), sizeOfValue(selectObjectPath(right.report().get("inference"), "nextTokens")));
            case "delta:combinedtext":
                return summarizeValueDelta("combinedText", selectObjectPath(left.report().get("inference"), "combinedText"), selectObjectPath(right.report().get("inference"), "combinedText"));
            case "delta:generatedtext":
                return summarizeValueDelta("generatedText", selectObjectPath(left.report().get("inference"), "generatedText"), selectObjectPath(right.report().get("inference"), "generatedText"));
            default:
                if (normalized.startsWith("left:")) {
                    return selectObjectPath(left.report(), section.substring("left:".length()));
                }
                if (normalized.startsWith("right:")) {
                    return selectObjectPath(right.report(), section.substring("right:".length()));
                }
                throw new IllegalArgumentException("Unknown section '" + section + "'. Use one of: overview, comparison, artifactSummary, artifactSummaryLine, compareSummary, compareSummaryLine, commands, commands:ci, selectors, selectors:ci, inspectfields, inspectfields:ci, status, health, ci, status:ci, health:ci, runs, left, right, delta:latestTrainLoss, delta:bestValidationLoss, delta:epochCount, delta:nextToken, delta:nextTokens, delta:nextTokensPreview, delta:nextTokensCount, delta:combinedText, delta:generatedText, result=standard, left:<path>, or right:<path>.");
        }
    }

    private static Object selectBundle(BundleInput bundle, String section) {
        String normalized;
        normalized = section == null ? "bundlesummary" : section.toLowerCase(Locale.ROOT);
        switch (normalized) {
            case "reportversion":
                return bundle.manifest().getOrDefault("reportVersion", COMPARE_BUNDLE_REPORT_VERSION);
            case "artifactsummary":
                return summarizeBundleArtifactSummary(bundle);
            case "artifactsummaryline":
                return summarizeBundleArtifactSummaryLine(bundle);
            case "bundlesummary":
            case "overview":
            case "bundleoverview":
                return summarizeBundle(bundle);
            case "comparisonsummary":
                return bundle.manifest().getOrDefault("comparisonSummary", bundleManifestComparisonSummary(asMap(bundle.manifest().get("comparison"))));
            case "bundlehealth":
            case "health":
                return summarizeBundleHealth(bundle);
            case "bundlestatus":
                return summarizeBundleHealthShort(bundle);
            case "bundlemanifest":
            case "manifest":
                return bundle.manifest();
            case "bundlefiles":
            case "files":
                return selectBundleFiles(bundle, Map.of());
            case "commands":
                return summarizeBundleCommands(bundle, false);
            case "commands:ci":
                return summarizeBundleCommands(bundle, true);
            case "selectors":
                return summarizeBundleSelectors(bundle, false);
            case "selectors:ci":
                return summarizeBundleSelectors(bundle, true);
            case "inspectfields":
                return summarizeBundleInspectFields(bundle, false);
            case "inspectfields:ci":
                return summarizeBundleInspectFields(bundle, true);
            case "status":
            case "ci":
            case "status:ci":
                return summarizeBundleHealthShort(bundle);
            default:
                if (normalized.startsWith("bundleoverview:")) {
                    return summarizeBundleOverview(bundle, parseSelectorOptions(section.substring("bundleOverview:".length())));
                }
                if (normalized.startsWith("bundlehealth:")) {
                    return summarizeBundleHealth(bundle, parseSelectorOptions(section.substring("bundleHealth:".length())));
                }
                if (normalized.startsWith("filessummary:")) {
                    return summarizeBundleFiles(bundle, parseSelectorOptions(section.substring("filesSummary:".length())));
                }
                if (normalized.startsWith("bundlefiles:")) {
                    return selectBundleFiles(bundle, parseSelectorOptions(section.substring("bundleFiles:".length())));
                }
                if (normalized.startsWith("files:")) {
                    return selectBundleFiles(bundle, parseSelectorOptions(section.substring("files:".length())));
                }
                if (normalized.startsWith("commands:")) {
                    return summarizeBundleCommands(bundle, normalized.endsWith(":ci"));
                }
                if (normalized.startsWith("selectors:")) {
                    return summarizeBundleSelectors(bundle, normalized.endsWith(":ci"));
                }
                if (normalized.startsWith("inspectfields:")) {
                    return summarizeBundleInspectFields(bundle, normalized.endsWith(":ci"));
                }
                if (normalized.startsWith("loadfile:")) {
                    return loadBundleFile(bundle, section.substring("loadfile:".length()));
                }
                if (normalized.startsWith("file:")) {
                    return bundleFileEntry(bundle, section.substring("file:".length()));
                }
                throw new IllegalArgumentException("Unknown bundle section '" + section + "'. Use one of: reportVersion, artifactSummary, artifactSummaryLine, bundleSummary, comparisonSummary, bundleOverview, bundleOverview:ci, bundleStatus, bundleHealth, bundleHealth:ci, commands, commands:ci, selectors, selectors:ci, inspectfields, inspectfields:ci, bundleHealth:summary=short, bundleHealth:focus=files:section=runs:status=missing:top=5, bundleManifest, bundleFiles, files:section=<section>, files:name=<name>, files:format=json, files:sort=name, filesSummary:by=section, filesSummary:by=format, filesSummary:by=section:sort=-count:top=3, loadfile:<section>, loadfile:<name>, loadfile:section=<section>:pick=last, loadfile:section=<section>:index=0, or file:<name>.");
        }
    }

    private static Object selectResult(ResultInput result, String section) {
        String normalized = section == null ? "resultsummary" : section.toLowerCase(Locale.ROOT);
        switch (normalized) {
            case "reportversion":
                return result.result().getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION);
            case "resultoverview":
                return result.result().getOrDefault("resultOverview", result.result().getOrDefault("artifactSummary", Map.of()));
            case "resultoverview:ci":
                return result.result().getOrDefault("resultOverviewCi", result.result().getOrDefault("artifactSummaryLine", ""));
            case "resultmanifest":
                return result.result().getOrDefault("resultManifest", result.result());
            case "resultmanifest:ci":
                return result.result().getOrDefault("resultManifestCi", result.result().getOrDefault("resultSummaryLine", summarizeResultSummaryLine(result)));
            case "resultcomparison":
                return result.result().getOrDefault("resultComparison", result.result().getOrDefault("comparison", Map.of()));
            case "resultcomparison:ci":
                return result.result().getOrDefault("resultComparisonCi", result.result().getOrDefault("compareSummaryLine", ""));
            case "resultdashboard":
                return result.result().getOrDefault("resultDashboard", summarizeResultDashboard(result));
            case "resultdashboard:ci":
                return result.result().getOrDefault("resultDashboardCi", summarizeResultDashboardLine(result));
            case "resultdashboardmanifest":
                return result.result().getOrDefault("resultDashboardManifest", summarizeResultDashboardManifest(result));
            case "resultdashboardmanifest:ci":
                return result.result().getOrDefault("resultDashboardManifestCi", summarizeResultDashboardManifestCi(result));
            case "resultdashboardsummary":
                return result.result().getOrDefault("resultDashboardSummary", summarizeResultDashboardSummary(result));
            case "resultdashboardsummary:ci":
                return result.result().getOrDefault("resultDashboardSummaryCi", summarizeResultDashboardSummaryCi(result));
            case "resultdashboardoverview":
                return result.result().getOrDefault("resultDashboardOverview", summarizeResultDashboardOverview(result));
            case "resultdashboardoverview:ci":
                return result.result().getOrDefault("resultDashboardOverviewCi", summarizeResultDashboardOverviewCi(result));
            case "resultdashboardstatus":
                return result.result().getOrDefault("resultDashboardStatus", summarizeResultDashboardStatus(result));
            case "resultdashboardstatus:ci":
                return result.result().getOrDefault("resultDashboardStatusCi", summarizeResultDashboardStatusCi(result));
            case "resultdashboardhealth":
                return result.result().getOrDefault("resultDashboardHealth", summarizeResultDashboardHealth(result));
            case "resultdashboardhealth:ci":
                return result.result().getOrDefault("resultDashboardHealthCi", summarizeResultDashboardHealthCi(result));
            case "resultdashboardaliases":
                return result.result().getOrDefault("resultDashboardAliases", summarizeResultDashboardAliases(result));
            case "resultdashboardaliases:ci":
                return result.result().getOrDefault("resultDashboardAliasesCi", summarizeResultDashboardAliasesCi(result));
            case "resultdashboardtree":
                return result.result().getOrDefault("resultDashboardTree", summarizeResultDashboardTree(result));
            case "resultdashboardtree:ci":
                return result.result().getOrDefault("resultDashboardTreeCi", summarizeResultDashboardTreeCi(result));
            case "resultdashboardtreemanifest":
                return result.result().getOrDefault("resultDashboardTreeManifest", summarizeResultDashboardTreeManifest(result));
            case "resultdashboardtreemanifest:ci":
                return result.result().getOrDefault("resultDashboardTreeManifestCi", summarizeResultDashboardTreeManifestCi(result));
            case "resultdashboardtreesummary":
                return result.result().getOrDefault("resultDashboardTreeSummary", summarizeResultDashboardTreeSummary(result));
            case "resultdashboardtreesummary:ci":
                return result.result().getOrDefault("resultDashboardTreeSummaryCi", summarizeResultDashboardTreeSummaryCi(result));
            case "resultdashboardtreedashboard":
                return result.result().getOrDefault("resultDashboardTreeDashboard", summarizeResultDashboardTreeDashboard(result));
            case "resultdashboardtreedashboard:ci":
                return result.result().getOrDefault("resultDashboardTreeDashboardCi", summarizeResultDashboardTreeDashboardCi(result));
            case "resultdashboardtreepaths":
                return result.result().getOrDefault("resultDashboardTreePaths", summarizeResultDashboardTreePaths(result));
            case "resultdashboardtreepaths:ci":
                return result.result().getOrDefault("resultDashboardTreePathsCi", summarizeResultDashboardTreePathsCi(result));
            case "resultdashboardtreecatalog":
                return result.result().getOrDefault("resultDashboardTreeCatalog", summarizeResultDashboardTreeCatalog(result));
            case "resultdashboardtreecatalog:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogCi", summarizeResultDashboardTreeCatalogCi(result));
            case "resultdashboardtreecatalogmanifest":
                return result.result().getOrDefault("resultDashboardTreeCatalogManifest", summarizeResultDashboardTreeCatalogManifest(result));
            case "resultdashboardtreecatalogmanifest:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogManifestCi", summarizeResultDashboardTreeCatalogManifestCi(result));
            case "resultdashboardtreecatalogsummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogSummary", summarizeResultDashboardTreeCatalogSummary(result));
            case "resultdashboardtreecatalogsummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogSummaryCi", summarizeResultDashboardTreeCatalogSummaryCi(result));
            case "resultdashboardtreecatalogoverview":
                return result.result().getOrDefault("resultDashboardTreeCatalogOverview", summarizeResultDashboardTreeCatalogOverview(result));
            case "resultdashboardtreecatalogoverview:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogOverviewCi", summarizeResultDashboardTreeCatalogOverviewCi(result));
            case "resultdashboardtreecatalogtreesummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummary", summarizeResultDashboardTreeCatalogTreeSummary(result));
            case "resultdashboardtreecatalogtreesummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryCi", summarizeResultDashboardTreeCatalogTreeSummaryCi(result));
            case "resultdashboardtreecatalogtreesummaryfields":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryFields", summarizeResultDashboardTreeCatalogTreeSummaryFields(result));
            case "resultdashboardtreecatalogtreesummaryfields:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryFieldsCi", summarizeResultDashboardTreeCatalogTreeSummaryFieldsCi(result));
            case "resultdashboardtreecatalogtreesummarycommands":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryCommands", summarizeResultDashboardTreeCatalogTreeSummaryCommands(result));
            case "resultdashboardtreecatalogtreesummarycommands:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryCommandsCi", summarizeResultDashboardTreeCatalogTreeSummaryCommandsCi(result));
            case "resultdashboardtreecatalogtreesummaryselectors":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummarySelectors", summarizeResultDashboardTreeCatalogTreeSummarySelectors(result));
            case "resultdashboardtreecatalogtreesummaryselectors:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummarySelectorsCi", summarizeResultDashboardTreeCatalogTreeSummarySelectorsCi(result));
            case "resultdashboardtreecatalogtreesummaryaliases":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryAliases", summarizeResultDashboardTreeCatalogTreeSummaryAliases(result));
            case "resultdashboardtreecatalogtreesummaryaliases:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryAliasesCi", summarizeResultDashboardTreeCatalogTreeSummaryAliasesCi(result));
            case "resultdashboardtreecatalogtreesummarydiscovery":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryDiscovery", summarizeResultDashboardTreeCatalogTreeSummaryDiscovery(result));
            case "resultdashboardtreecatalogtreesummarydiscovery:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryDiscoveryCi", summarizeResultDashboardTreeCatalogTreeSummaryDiscoveryCi(result));
            case "resultdashboardtreecatalogtreesummarylineage":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryLineage", summarizeResultDashboardTreeCatalogTreeSummaryLineage(result));
            case "resultdashboardtreecatalogtreesummarylineage:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryLineageCi", summarizeResultDashboardTreeCatalogTreeSummaryLineageCi(result));
            case "resultdashboardtreecatalogtreesummarynavigation":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryNavigation", summarizeResultDashboardTreeCatalogTreeSummaryNavigation(result));
            case "resultdashboardtreecatalogtreesummarynavigation:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryNavigationCi", summarizeResultDashboardTreeCatalogTreeSummaryNavigationCi(result));
            case "resultdashboardtreecatalogtreesummarybreadcrumbs":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryBreadcrumbs", summarizeResultDashboardTreeCatalogTreeSummaryBreadcrumbs(result));
            case "resultdashboardtreecatalogtreesummarybreadcrumbs:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryBreadcrumbsCi", summarizeResultDashboardTreeCatalogTreeSummaryBreadcrumbsCi(result));
            case "resultdashboardtreecatalogtreesummarysitemap":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummarySitemap", summarizeResultDashboardTreeCatalogTreeSummarySitemap(result));
            case "resultdashboardtreecatalogtreesummarysitemap:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummarySitemapCi", summarizeResultDashboardTreeCatalogTreeSummarySitemapCi(result));
            case "resultdashboardtreecatalogtreesummaryindex":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryIndex", summarizeResultDashboardTreeCatalogTreeSummaryIndex(result));
            case "resultdashboardtreecatalogtreesummaryindex:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryIndexCi", summarizeResultDashboardTreeCatalogTreeSummaryIndexCi(result));
            case "resultdashboardtreecatalogtreesummaryregistry":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistry", summarizeResultDashboardTreeCatalogTreeSummaryRegistry(result));
            case "resultdashboardtreecatalogtreesummaryregistry:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryfields":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryFields", summarizeResultDashboardTreeCatalogTreeSummaryRegistryFields(result));
            case "resultdashboardtreecatalogtreesummaryregistryfields:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryFieldsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryFieldsCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrycommands":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryCommands", summarizeResultDashboardTreeCatalogTreeSummaryRegistryCommands(result));
            case "resultdashboardtreecatalogtreesummaryregistrycommands:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryCommandsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryCommandsCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryselectors":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySelectors", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySelectors(result));
            case "resultdashboardtreecatalogtreesummaryregistryselectors:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySelectorsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySelectorsCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryaliases":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryAliases", summarizeResultDashboardTreeCatalogTreeSummaryRegistryAliases(result));
            case "resultdashboardtreecatalogtreesummaryregistryaliases:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryAliasesCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryAliasesCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrydiscovery":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryDiscovery", summarizeResultDashboardTreeCatalogTreeSummaryRegistryDiscovery(result));
            case "resultdashboardtreecatalogtreesummaryregistrydiscovery:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryDiscoveryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrynavigation":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryNavigation", summarizeResultDashboardTreeCatalogTreeSummaryRegistryNavigation(result));
            case "resultdashboardtreecatalogtreesummaryregistrynavigation:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryNavigationCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryNavigationCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrybreadcrumbs":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs", summarizeResultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs(result));
            case "resultdashboardtreecatalogtreesummaryregistrybreadcrumbs:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrysitemap":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySitemap", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySitemap(result));
            case "resultdashboardtreecatalogtreesummaryregistrysitemap:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySitemapCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySitemapCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryindex":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryIndex", summarizeResultDashboardTreeCatalogTreeSummaryRegistryIndex(result));
            case "resultdashboardtreecatalogtreesummaryregistryindex:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryIndexCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryIndexCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryoverview":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryOverview", summarizeResultDashboardTreeCatalogTreeSummaryRegistryOverview(result));
            case "resultdashboardtreecatalogtreesummaryregistryoverview:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryOverviewCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryOverviewCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrystatus":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryStatus", summarizeResultDashboardTreeCatalogTreeSummaryRegistryStatus(result));
            case "resultdashboardtreecatalogtreesummaryregistrystatus:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryStatusCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryStatusCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryhealth":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryHealth", summarizeResultDashboardTreeCatalogTreeSummaryRegistryHealth(result));
            case "resultdashboardtreecatalogtreesummaryregistryhealth:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryHealthCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryHealthCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrysummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySummary", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySummary(result));
            case "resultdashboardtreecatalogtreesummaryregistrysummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySummaryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySummaryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrymanifest":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryManifest", summarizeResultDashboardTreeCatalogTreeSummaryRegistryManifest(result));
            case "resultdashboardtreecatalogtreesummaryregistrymanifest:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryManifestCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryManifestCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryenvelope":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryEnvelope", summarizeResultDashboardTreeCatalogTreeSummaryRegistryEnvelope(result));
            case "resultdashboardtreecatalogtreesummaryregistryenvelope:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryEnvelopeCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrysnapshot":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySnapshot", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySnapshot(result));
            case "resultdashboardtreecatalogtreesummaryregistrysnapshot:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySnapshotCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistrySnapshotCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryaudit":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryAudit", summarizeResultDashboardTreeCatalogTreeSummaryRegistryAudit(result));
            case "resultdashboardtreecatalogtreesummaryregistryaudit:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryAuditCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryAuditCi(result));
            case "resultdashboardtreecatalogtreesummaryregistryreport":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryReport", summarizeResultDashboardTreeCatalogTreeSummaryRegistryReport(result));
            case "resultdashboardtreecatalogtreesummaryregistryreport:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryReportCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryReportCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklist":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklist", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklist(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklist:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistsummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistsummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrycheckliststatus":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus(result));
            case "resultdashboardtreecatalogtreesummaryregistrycheckliststatus:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklisthealth":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklisthealth:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistbadge":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistbadge:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistcard":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistcard:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistCardCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistpanel":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistpanel:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistwidget":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistwidget:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklisttile":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTile(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklisttile:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTileCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistmetric":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistmetric:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklisttrend":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklisttrend:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistgauge":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistgauge:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistsummarycard":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistsummarycard:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistactionbar":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistactionbar:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboard":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboard:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardmanifest":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardmanifest:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardsnapshot":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardsnapshot:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardaudit":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardaudit:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardreport":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardreport:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardenvelope":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardenvelope:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardpackage":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardpackage:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardbundle":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardbundle:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboarddelivery":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboarddelivery:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliveryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliveryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardreceipt":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardreceipt:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardarchive":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardarchive:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardledger":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardledger:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardhistory":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardhistory:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistoryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistoryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardtimeline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardtimeline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardtrace":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardtrace:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardtelemetry":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardtelemetry:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardobservation":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardobservation:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardsignal":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardsignal:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardinsight":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardinsight:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardrecommendation":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardrecommendation:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboarddecision":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboarddecision:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgatejson":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgatejson:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateenv":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateenv:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgatemarkdown":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgatemarkdown:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgatesummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgatesummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateartifact":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateartifact:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadmanifest":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadmanifest:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadscript":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadscript:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadreadme":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadreadme:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadbundle":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadbundle:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadindex":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadindex:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesindex":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesindex:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesjsonline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesjsonline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilescsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilescsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilestsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilestsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesenv":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesenv:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnvCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesmarkdown":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfilesmarkdown:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdownCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksums":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksums:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsbadge":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsbadge:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadgeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumscard":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumscard:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCardCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumspanel":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumspanel:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanelCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumswidget":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumswidget:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidgetCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstile":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstile:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTileCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsmetric":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsmetric:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetricCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstrend":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstrend:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrendCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsgauge":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsgauge:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGaugeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssummarycard":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssummarycard:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCardCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsactionbar":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsactionbar:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBarCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsmenu":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsmenu:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenuCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstoolbar":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstoolbar:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbarCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumscommandpalette":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumscommandpalette:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPaletteCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssearchindex":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssearchindex:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndexCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssearchresults":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumssearchresults:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResultsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsfilterstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsfilterstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsviewstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsviewstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsemptystate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsemptystate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsloadingstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsloadingstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumserrorstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumserrorstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstoaststate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstoaststate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationcenter":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationcenter:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenterCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationbadge":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationbadge:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadgeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanel":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanel:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneljsonline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneljsonline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelcsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelcsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneltsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneltsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelmarkdown":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelmarkdown:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdownCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelenv":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelenv:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnvCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelsummarycard":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelsummarycard:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCardCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelactionbar":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelactionbar:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBarCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelmenu":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelmenu:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenuCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneltoolbar":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneltoolbar:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbarCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelcommandpalette":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelcommandpalette:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPaletteCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelsearchindex":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelsearchindex:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndexCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelsearchresults":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelsearchresults:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResultsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelfilterstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelfilterstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelviewstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelviewstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelemptystate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelemptystate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelloadingstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelloadingstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelerrorstate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelerrorstate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneltoaststate":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpaneltoaststate:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastStateCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationcenter":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationcenter:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenterCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationbadge":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationbadge:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadgeCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationjsonline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsnotificationpanelnotificationjsonline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsjsonline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsjsonline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumscsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumscsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumstsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsmarkdown":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsmarkdown:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdownCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsenv":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadfileschecksumsenv:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnvCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadstatus":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadstatus:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadstatusline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadstatusline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadenv":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadenv:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnvCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadjsonline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadjsonline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadcsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadcsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadtsvline":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdashboardgateuploadtsvline:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLineCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdetails":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails(result));
            case "resultdashboardtreecatalogtreesummaryregistrychecklistdetails:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsCi", summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsCi(result));
            case "resultdashboardtreecatalogfields":
                return result.result().getOrDefault("resultDashboardTreeCatalogFields", summarizeResultDashboardTreeCatalogFields(result));
            case "resultdashboardtreecatalogfields:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogFieldsCi", summarizeResultDashboardTreeCatalogFieldsCi(result));
            case "resultdashboardtreecatalogcommands":
                return result.result().getOrDefault("resultDashboardTreeCatalogCommands", summarizeResultDashboardTreeCatalogCommands(result));
            case "resultdashboardtreecatalogcommands:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogCommandsCi", summarizeResultDashboardTreeCatalogCommandsCi(result));
            case "resultdashboardtreecatalogcommandssummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogCommandsSummary", summarizeResultDashboardTreeCatalogCommandsSummary(result));
            case "resultdashboardtreecatalogcommandssummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogCommandsSummaryCi", summarizeResultDashboardTreeCatalogCommandsSummaryCi(result));
            case "resultdashboardtreecatalogselectors":
                return result.result().getOrDefault("resultDashboardTreeCatalogSelectors", summarizeResultDashboardTreeCatalogSelectors(result));
            case "resultdashboardtreecatalogselectors:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogSelectorsCi", summarizeResultDashboardTreeCatalogSelectorsCi(result));
            case "resultdashboardtreecatalogselectorssummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogSelectorsSummary", summarizeResultDashboardTreeCatalogSelectorsSummary(result));
            case "resultdashboardtreecatalogselectorssummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogSelectorsSummaryCi", summarizeResultDashboardTreeCatalogSelectorsSummaryCi(result));
            case "resultdashboardtreecatalogaliases":
                return result.result().getOrDefault("resultDashboardTreeCatalogAliases", summarizeResultDashboardTreeCatalogAliases(result));
            case "resultdashboardtreecatalogaliases:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogAliasesCi", summarizeResultDashboardTreeCatalogAliasesCi(result));
            case "resultdashboardtreecatalogaliasessummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogAliasesSummary", summarizeResultDashboardTreeCatalogAliasesSummary(result));
            case "resultdashboardtreecatalogaliasessummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogAliasesSummaryCi", summarizeResultDashboardTreeCatalogAliasesSummaryCi(result));
            case "resultdashboardtreecatalogchildrensummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogChildrenSummary", summarizeResultDashboardTreeCatalogChildrenSummary(result));
            case "resultdashboardtreecatalogchildrensummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogChildrenSummaryCi", summarizeResultDashboardTreeCatalogChildrenSummaryCi(result));
            case "resultdashboardtreecatalogcountssummary":
                return result.result().getOrDefault("resultDashboardTreeCatalogCountsSummary", summarizeResultDashboardTreeCatalogCountsSummary(result));
            case "resultdashboardtreecatalogcountssummary:ci":
                return result.result().getOrDefault("resultDashboardTreeCatalogCountsSummaryCi", summarizeResultDashboardTreeCatalogCountsSummaryCi(result));
            case "resultdashboardtreechildren":
                return result.result().getOrDefault("resultDashboardTreeChildren", summarizeResultDashboardTreeChildren(result));
            case "resultdashboardtreechildren:ci":
                return result.result().getOrDefault("resultDashboardTreeChildrenCi", summarizeResultDashboardTreeChildrenCi(result));
            case "resultdashboardtreecounts":
                return result.result().getOrDefault("resultDashboardTreeCounts", summarizeResultDashboardTreeCounts(result));
            case "resultdashboardtreecounts:ci":
                return result.result().getOrDefault("resultDashboardTreeCountsCi", summarizeResultDashboardTreeCountsCi(result));
            case "resultdashboardtreefields":
                return result.result().getOrDefault("resultDashboardTreeFields", summarizeResultDashboardTreeFields(result));
            case "resultdashboardtreefields:ci":
                return result.result().getOrDefault("resultDashboardTreeFieldsCi", summarizeResultDashboardTreeFieldsCi(result));
            case "resultdashboardtreealiases":
                return result.result().getOrDefault("resultDashboardTreeAliases", summarizeResultDashboardTreeAliases(result));
            case "resultdashboardtreealiases:ci":
                return result.result().getOrDefault("resultDashboardTreeAliasesCi", summarizeResultDashboardTreeAliasesCi(result));
            case "resultdashboardtreecommands":
                return result.result().getOrDefault("resultDashboardTreeCommands", summarizeResultDashboardTreeCommands(result));
            case "resultdashboardtreecommands:ci":
                return result.result().getOrDefault("resultDashboardTreeCommandsCi", summarizeResultDashboardTreeCommandsCi(result));
            case "resultdashboardtreeselectors":
                return result.result().getOrDefault("resultDashboardTreeSelectors", summarizeResultDashboardTreeSelectors(result));
            case "resultdashboardtreeselectors:ci":
                return result.result().getOrDefault("resultDashboardTreeSelectorsCi", summarizeResultDashboardTreeSelectorsCi(result));
            case "resultdashboardfamilyfields":
                return result.result().getOrDefault("resultDashboardFamilyFields", summarizeResultDashboardFamilyFields(result));
            case "resultdashboardfamilyfields:ci":
                return result.result().getOrDefault("resultDashboardFamilyFieldsCi", summarizeResultDashboardFamilyFieldsCi(result));
            case "resultdashboardfamilycommands":
                return result.result().getOrDefault("resultDashboardFamilyCommands", summarizeResultDashboardFamilyCommands(result));
            case "resultdashboardfamilycommands:ci":
                return result.result().getOrDefault("resultDashboardFamilyCommandsCi", summarizeResultDashboardFamilyCommandsCi(result));
            case "resultdashboardfamilyselectors":
                return result.result().getOrDefault("resultDashboardFamilySelectors", summarizeResultDashboardFamilySelectors(result));
            case "resultdashboardfamilyselectors:ci":
                return result.result().getOrDefault("resultDashboardFamilySelectorsCi", summarizeResultDashboardFamilySelectorsCi(result));
            case "resultdashboardselectors":
                return result.result().getOrDefault("resultDashboardSelectors", summarizeResultDashboardSelectors(result));
            case "resultdashboardselectors:ci":
                return result.result().getOrDefault("resultDashboardSelectorsCi", summarizeResultDashboardSelectorsCi(result));
            case "resultdashboardcommands":
                return result.result().getOrDefault("resultDashboardCommands", summarizeResultDashboardCommands(result));
            case "resultdashboardcommands:ci":
                return result.result().getOrDefault("resultDashboardCommandsCi", summarizeResultDashboardCommandsCi(result));
            case "resultdashboardfields":
                return result.result().getOrDefault("resultDashboardFields", summarizeResultDashboardFields(result));
            case "resultdashboardfields:ci":
                return result.result().getOrDefault("resultDashboardFieldsCi", summarizeResultDashboardFieldsCi(result));
            case "resultpayloads":
                return summarizeResultPayloads(result);
            case "resultpayloads:ci":
                return summarizeResultPayloadsLine(result);
            case "resultsources":
                return summarizeResultPayloads(result);
            case "resultsources:ci":
                return summarizeResultPayloadsLine(result);
            case "resultaliases":
                return summarizeResultAliases(result, false);
            case "resultaliases:ci":
                return summarizeResultAliases(result, true);
            case "resultstatus":
                return result.result().getOrDefault("resultStatus", result.result().getOrDefault("compareSummary", Map.of()));
            case "resultstatus:ci":
                return result.result().getOrDefault("resultStatusCi", result.result().getOrDefault("compareSummaryLine", ""));
            case "resulthealth":
                return result.result().getOrDefault("resultHealth", result.result().getOrDefault("compareSummary", Map.of()));
            case "resulthealth:ci":
                return result.result().getOrDefault("resultHealthCi", result.result().getOrDefault("compareSummaryLine", ""));
            case "resultsummary":
                return summarizeResultSummary(result);
            case "resultsummaryline":
                return summarizeResultSummaryLine(result);
            case "resultcommands":
                return summarizeResultCommands(result, false);
            case "resultcommands:ci":
                return summarizeResultCommands(result, true);
            case "resultselectors":
                return summarizeResultSelectors(result, false);
            case "resultselectors:ci":
                return summarizeResultSelectors(result, true);
            case "resultfields":
                return summarizeResultFields(result, false);
            case "resultfields:ci":
                return summarizeResultFields(result, true);
            case "artifactsummary":
                return result.result().getOrDefault("artifactSummary", Map.of());
            case "artifactsummaryline":
                return result.result().getOrDefault("artifactSummaryLine", "");
            case "comparison":
                return result.result().getOrDefault("comparison", Map.of());
            case "comparesummary":
                return result.result().getOrDefault("compareSummary", Map.of());
            case "comparesummaryline":
                return result.result().getOrDefault("compareSummaryLine", "");
            case "commands":
                return result.result().getOrDefault("commands", result.result().getOrDefault("commandsCi", Map.of()));
            case "commands:ci":
                return result.result().getOrDefault("commandsCi", Map.of());
            case "selectors":
                return result.result().getOrDefault("selectors", result.result().getOrDefault("selectorsCi", Map.of()));
            case "selectors:ci":
                return result.result().getOrDefault("selectorsCi", Map.of());
            case "inspectfields":
                return result.result().getOrDefault("inspectfields", result.result().getOrDefault("inspectfieldsCi", Map.of()));
            case "inspectfields:ci":
                return result.result().getOrDefault("inspectfieldsCi", Map.of());
            case "manifest":
            case "result":
                return result.result();
            default:
                throw new IllegalArgumentException("Unknown result section '" + section + "'. Use one of: reportVersion, resultOverview, resultOverview:ci, resultManifest, resultManifest:ci, resultComparison, resultComparison:ci, resultDashboard, resultDashboard:ci, resultDashboardManifest, resultDashboardManifest:ci, resultDashboardSummary, resultDashboardSummary:ci, resultDashboardOverview, resultDashboardOverview:ci, resultDashboardStatus, resultDashboardStatus:ci, resultDashboardHealth, resultDashboardHealth:ci, resultDashboardAliases, resultDashboardAliases:ci, resultDashboardTree, resultDashboardTree:ci, resultDashboardTreeManifest, resultDashboardTreeManifest:ci, resultDashboardTreeSummary, resultDashboardTreeSummary:ci, resultDashboardTreeDashboard, resultDashboardTreeDashboard:ci, resultDashboardTreePaths, resultDashboardTreePaths:ci, resultDashboardTreeCatalog, resultDashboardTreeCatalog:ci, resultDashboardTreeCatalogManifest, resultDashboardTreeCatalogManifest:ci, resultDashboardTreeCatalogSummary, resultDashboardTreeCatalogSummary:ci, resultDashboardTreeCatalogOverview, resultDashboardTreeCatalogOverview:ci, resultDashboardTreeCatalogTreeSummary, resultDashboardTreeCatalogTreeSummary:ci, resultDashboardTreeCatalogTreeSummaryFields, resultDashboardTreeCatalogTreeSummaryFields:ci, resultDashboardTreeCatalogTreeSummaryCommands, resultDashboardTreeCatalogTreeSummaryCommands:ci, resultDashboardTreeCatalogTreeSummarySelectors, resultDashboardTreeCatalogTreeSummarySelectors:ci, resultDashboardTreeCatalogTreeSummaryAliases, resultDashboardTreeCatalogTreeSummaryAliases:ci, resultDashboardTreeCatalogTreeSummaryDiscovery, resultDashboardTreeCatalogTreeSummaryDiscovery:ci, resultDashboardTreeCatalogTreeSummaryLineage, resultDashboardTreeCatalogTreeSummaryLineage:ci, resultDashboardTreeCatalogTreeSummaryNavigation, resultDashboardTreeCatalogTreeSummaryNavigation:ci, resultDashboardTreeCatalogTreeSummaryBreadcrumbs, resultDashboardTreeCatalogTreeSummaryBreadcrumbs:ci, resultDashboardTreeCatalogTreeSummarySitemap, resultDashboardTreeCatalogTreeSummarySitemap:ci, resultDashboardTreeCatalogTreeSummaryIndex, resultDashboardTreeCatalogTreeSummaryIndex:ci, resultDashboardTreeCatalogTreeSummaryRegistry, resultDashboardTreeCatalogTreeSummaryRegistry:ci, resultDashboardTreeCatalogTreeSummaryRegistryFields, resultDashboardTreeCatalogTreeSummaryRegistryFields:ci, resultDashboardTreeCatalogTreeSummaryRegistryCommands, resultDashboardTreeCatalogTreeSummaryRegistryCommands:ci, resultDashboardTreeCatalogTreeSummaryRegistrySelectors, resultDashboardTreeCatalogTreeSummaryRegistrySelectors:ci, resultDashboardTreeCatalogTreeSummaryRegistryAliases, resultDashboardTreeCatalogTreeSummaryRegistryAliases:ci, resultDashboardTreeCatalogTreeSummaryRegistryDiscovery, resultDashboardTreeCatalogTreeSummaryRegistryDiscovery:ci, resultDashboardTreeCatalogTreeSummaryRegistryNavigation, resultDashboardTreeCatalogTreeSummaryRegistryNavigation:ci, resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs, resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs:ci, resultDashboardTreeCatalogTreeSummaryRegistrySitemap, resultDashboardTreeCatalogTreeSummaryRegistrySitemap:ci, resultDashboardTreeCatalogTreeSummaryRegistryIndex, resultDashboardTreeCatalogTreeSummaryRegistryIndex:ci, resultDashboardTreeCatalogTreeSummaryRegistryOverview, resultDashboardTreeCatalogTreeSummaryRegistryOverview:ci, resultDashboardTreeCatalogTreeSummaryRegistryStatus, resultDashboardTreeCatalogTreeSummaryRegistryStatus:ci, resultDashboardTreeCatalogTreeSummaryRegistryHealth, resultDashboardTreeCatalogTreeSummaryRegistryHealth:ci, resultDashboardTreeCatalogTreeSummaryRegistrySummary, resultDashboardTreeCatalogTreeSummaryRegistrySummary:ci, resultDashboardTreeCatalogTreeSummaryRegistryManifest, resultDashboardTreeCatalogTreeSummaryRegistryManifest:ci, resultDashboardTreeCatalogTreeSummaryRegistryEnvelope, resultDashboardTreeCatalogTreeSummaryRegistryEnvelope:ci, resultDashboardTreeCatalogTreeSummaryRegistrySnapshot, resultDashboardTreeCatalogTreeSummaryRegistrySnapshot:ci, resultDashboardTreeCatalogTreeSummaryRegistryAudit, resultDashboardTreeCatalogTreeSummaryRegistryAudit:ci, resultDashboardTreeCatalogTreeSummaryRegistryReport, resultDashboardTreeCatalogTreeSummaryRegistryReport:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklist, resultDashboardTreeCatalogTreeSummaryRegistryChecklist:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary, resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus, resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth, resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge, resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard, resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel, resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget, resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile, resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric, resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend, resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge, resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard, resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar, resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine:ci, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails, resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci, resultDashboardTreeCatalogFields, resultDashboardTreeCatalogFields:ci, resultDashboardTreeCatalogCommands, resultDashboardTreeCatalogCommands:ci, resultDashboardTreeCatalogCommandsSummary, resultDashboardTreeCatalogCommandsSummary:ci, resultDashboardTreeCatalogSelectors, resultDashboardTreeCatalogSelectors:ci, resultDashboardTreeCatalogSelectorsSummary, resultDashboardTreeCatalogSelectorsSummary:ci, resultDashboardTreeCatalogAliases, resultDashboardTreeCatalogAliases:ci, resultDashboardTreeCatalogAliasesSummary, resultDashboardTreeCatalogAliasesSummary:ci, resultDashboardTreeCatalogChildrenSummary, resultDashboardTreeCatalogChildrenSummary:ci, resultDashboardTreeCatalogCountsSummary, resultDashboardTreeCatalogCountsSummary:ci, resultDashboardTreeChildren, resultDashboardTreeChildren:ci, resultDashboardTreeCounts, resultDashboardTreeCounts:ci, resultDashboardTreeFields, resultDashboardTreeFields:ci, resultDashboardTreeAliases, resultDashboardTreeAliases:ci, resultDashboardTreeCommands, resultDashboardTreeCommands:ci, resultDashboardTreeSelectors, resultDashboardTreeSelectors:ci, resultDashboardFamilyFields, resultDashboardFamilyFields:ci, resultDashboardFamilyCommands, resultDashboardFamilyCommands:ci, resultDashboardFamilySelectors, resultDashboardFamilySelectors:ci, resultDashboardSelectors, resultDashboardSelectors:ci, resultDashboardCommands, resultDashboardCommands:ci, resultDashboardFields, resultDashboardFields:ci, resultPayloads, resultPayloads:ci, resultSources, resultSources:ci, resultAliases, resultAliases:ci, resultStatus, resultStatus:ci, resultHealth, resultHealth:ci, resultSummary, resultSummaryLine, resultCommands, resultCommands:ci, resultSelectors, resultSelectors:ci, resultFields, resultFields:ci, artifactSummary, artifactSummaryLine, comparison, compareSummary, compareSummaryLine, commands, commands:ci, selectors, selectors:ci, inspectfields, inspectfields:ci, or result.");
        }
    }

    private static Map<String, Object> writeBundle(InputBundle left, InputBundle right, String bundleSection, Path outputDir) {
        if (!"bundle=standard".equals(bundleSection)) {
            throw new IllegalArgumentException("Unknown bundle mode '" + bundleSection + "'. Use bundle=standard.");
        }
        try {
            Path bundleDir = outputDir.toAbsolutePath().normalize();
            Files.createDirectories(bundleDir, new FileAttribute[0]);
            Map<String, Object> overview = summarizeOverview(left, right);
            List<Map<String, Object>> runs = List.of(summarizeRun("left", left), summarizeRun("right", right));
            Map<String, Object> deltaLatestTrainLoss = summarizeNumericDelta("latestTrainLoss", asDouble(selectObjectPath(left.report().get("training"), "latestTrainLoss")), asDouble(selectObjectPath(right.report().get("training"), "latestTrainLoss")));
            Map<String, Object> deltaNextToken = summarizeValueDelta("nextToken", selectObjectPath(left.report().get("inference"), "nextToken"), selectObjectPath(right.report().get("inference"), "nextToken"));
            Map<String, Object> deltaNextTokens = summarizeValueDelta("nextTokens", selectObjectPath(left.report().get("inference"), "nextTokens"), selectObjectPath(right.report().get("inference"), "nextTokens"));
            Map<String, Object> deltaNextTokensPreview = summarizeValueDelta("nextTokensPreview", readNextTokensPreview(asMap(left.report().get("inference"))), readNextTokensPreview(asMap(right.report().get("inference"))));
            Map<String, Object> deltaNextTokensCount = summarizeNumericDelta("nextTokensCount", sizeOfValue(selectObjectPath(left.report().get("inference"), "nextTokens")), sizeOfValue(selectObjectPath(right.report().get("inference"), "nextTokens")));
            Map<String, Object> deltaCombinedText = summarizeValueDelta("combinedText", selectObjectPath(left.report().get("inference"), "combinedText"), selectObjectPath(right.report().get("inference"), "combinedText"));
            Path overviewFile = bundleDir.resolve("overview.json");
            Path runsFile = bundleDir.resolve("runs.json");
            Path deltaLatestTrainLossFile = bundleDir.resolve("delta-latestTrainLoss.json");
            Path deltaNextTokenFile = bundleDir.resolve("delta-nextToken.json");
            Path deltaNextTokensFile = bundleDir.resolve("delta-nextTokens.json");
            Path deltaNextTokensPreviewFile = bundleDir.resolve("delta-nextTokensPreview.json");
            Path deltaNextTokensCountFile = bundleDir.resolve("delta-nextTokensCount.json");
            Path deltaCombinedTextFile = bundleDir.resolve("delta-combinedText.json");
            JSON.writeValue(overviewFile.toFile(), overview);
            JSON.writeValue(runsFile.toFile(), runs);
            JSON.writeValue(deltaLatestTrainLossFile.toFile(), deltaLatestTrainLoss);
            JSON.writeValue(deltaNextTokenFile.toFile(), deltaNextToken);
            JSON.writeValue(deltaNextTokensFile.toFile(), deltaNextTokens);
            JSON.writeValue(deltaNextTokensPreviewFile.toFile(), deltaNextTokensPreview);
            JSON.writeValue(deltaNextTokensCountFile.toFile(), deltaNextTokensCount);
            JSON.writeValue(deltaCombinedTextFile.toFile(), deltaCombinedText);
            List<Map<String, Object>> files = List.of(bundleFileRow("overview", overviewFile), bundleFileRow("runs", runsFile), bundleFileRow("delta-latestTrainLoss", deltaLatestTrainLossFile), bundleFileRow("delta-nextToken", deltaNextTokenFile), bundleFileRow("delta-nextTokens", deltaNextTokensFile), bundleFileRow("delta-nextTokensPreview", deltaNextTokensPreviewFile), bundleFileRow("delta-nextTokensCount", deltaNextTokensCountFile), bundleFileRow("delta-combinedText", deltaCombinedTextFile));
            Map<String, Object> manifest = new LinkedHashMap<>();
            manifest.put("reportVersion", COMPARE_BUNDLE_REPORT_VERSION);
            manifest.put("bundleType", "byte-latent-train-infer-compare");
            manifest.put("bundleMode", "standard");
            manifest.put("bundleDir", bundleDir.toString());
            manifest.put("leftReportFile", left.reportFile().toAbsolutePath().normalize().toString());
            manifest.put("rightReportFile", right.reportFile().toAbsolutePath().normalize().toString());
            manifest.put("leftReportVersion", left.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
            manifest.put("rightReportVersion", right.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
            manifest.put("fileCount", Integer.valueOf(files.size()));
            manifest.put("files", files);
            Map<String, Object> comparison = comparisonBlock(left, right);
            manifest.put("comparison", comparison);
            manifest.put("leftNextTokensPreview", comparison.getOrDefault("leftNextTokensPreview", List.of()));
            manifest.put("rightNextTokensPreview", comparison.getOrDefault("rightNextTokensPreview", List.of()));
            manifest.put("nextTokensPreviewChanged", comparison.getOrDefault("nextTokensPreviewChanged", false));
            manifest.put("comparisonSummary", bundleManifestComparisonSummary(comparison));
            Map<String, Object> manifestStatus = summarizeBundleManifestStatus(files, bundleDir, manifest);
            manifest.put("status", manifestStatus);
            manifest.put("healthStatus", manifestStatus.getOrDefault("healthStatus", ""));
            manifest.put("missingFileCount", manifestStatus.getOrDefault("missingFileCount", 0));
            manifest.put("okFileCount", manifestStatus.getOrDefault("okFileCount", 0));
            manifest.put("statusSummary", bundleManifestStatusSummary(manifestStatus));
            manifest.put("inspectBundleStatus", bundleCommandText("bundleStatus", bundleDir.toString()));
            manifest.put("inspectBundleHealth", bundleCommandText("bundleHealth:ci", bundleDir.toString()));
            manifest.put("inspectBundleOverview", bundleCommandText("bundleOverview:ci", bundleDir.toString()));
            manifest.put("inspectBundleFiles", bundleCommandText("files:section=runs:sort=name", bundleDir.toString()));
            manifest.put("inspectBundleLoadfile", bundleCommandText("loadfile:section=runs:pick=last", bundleDir.toString()));
            manifest.put("inspectBundleSummarySelector", "bundleSummary");
            manifest.put("inspectBundleFilesSummarySelector", "filesSummary:by=section:sort=-count");
            manifest.put("inspectBundleNextTokensSelector", "loadfile:delta-nextTokens");
            manifest.put("inspectBundleNextTokensPreviewSelector", "loadfile:delta-nextTokensPreview");
            manifest.put("inspectBundleNextTokensCountSelector", "loadfile:delta-nextTokensCount");
            manifest.put("inspectBundleMissingFilesSelector", "bundleHealth:focus=files:status=missing");
            manifest.put("inspectBundleMissingFilesTopSelector", "bundleHealth:focus=files:status=missing:top=5");
            manifest.put("commands", bundleCommandRows(bundleDir.toString()));
            manifest.put("commandsCi", bundleCommandSummary(bundleDir.toString()));
            manifest.put("inspectCommands", bundleCommandRows(bundleDir.toString()));
            manifest.put("inspectCommandsCi", bundleCommandSummary(bundleDir.toString()));
            manifest.put("selectors", bundleSelectorRows(bundleDir.toString()));
            manifest.put("selectorsCi", bundleSelectorSummary(bundleDir.toString()));
            manifest.put("inspectfields", bundleInspectFieldRows());
            manifest.put("inspectfieldsCi", bundleInspectFieldSummary());
            manifest.put("inspectFields", bundleInspectFieldRows());
            manifest.put("inspectFieldsCi", bundleInspectFieldSummary());
            Map<String, Object> artifactSummary = bundleArtifactSummary(bundleDir.toString(), manifest);
            manifest.put("artifactSummary", artifactSummary);
            manifest.put("artifactSummaryLine", bundleArtifactSummaryLine(artifactSummary));
            Path manifestFile = bundleDir.resolve(BUNDLE_MANIFEST_FILE_NAME);
            JSON.writeValue(manifestFile.toFile(), manifest);
            manifest.put("manifestFile", manifestFile.toString());
            return manifest;
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to write bundle to " + String.valueOf(outputDir) + ".", exception);
        }
    }

    private static Map<String, Object> writeResult(InputBundle left, InputBundle right, String resultSection, Path outputFile) {
        if (!"result=standard".equals(resultSection)) {
            throw new IllegalArgumentException("Unknown result mode '" + resultSection + "'. Use result=standard.");
        }
        try {
            Path resultFile = outputFile.toAbsolutePath().normalize();
            Path parent = resultFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent, new FileAttribute[0]);
            }
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("reportVersion", COMPARE_RESULT_REPORT_VERSION);
            result.put("leftReportVersion", left.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
            result.put("rightReportVersion", right.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
            result.put("leftReportFile", left.reportFile().toAbsolutePath().normalize().toString());
            result.put("rightReportFile", right.reportFile().toAbsolutePath().normalize().toString());
            result.put("artifactSummary", compareArtifactSummary(left, right));
            result.put("artifactSummaryLine", compareArtifactSummaryLine(left, right));
            result.put("comparison", comparisonBlock(left, right));
            result.put("compareSummary", compareSummary(left, right));
            result.put("compareSummaryLine", compareSummaryLine(left, right));
            result.put("resultOverview", result.get("artifactSummary"));
            result.put("resultOverviewCi", result.get("artifactSummaryLine"));
            result.put("resultComparison", result.get("comparison"));
            result.put("resultComparisonCi", result.get("compareSummaryLine"));
            result.put("resultStatus", result.get("compareSummary"));
            result.put("resultStatusCi", result.get("compareSummaryLine"));
            result.put("resultHealth", result.get("compareSummary"));
            result.put("resultHealthCi", result.get("compareSummaryLine"));
            result.put("commands", compareCommands(left, right, false));
            result.put("commandsCi", compareCommands(left, right, true));
            result.put("resultCommands", resultCommandRows(resultFile.toString()));
            result.put("resultCommandsCi", resultCommandSummary(resultFile.toString()));
            result.put("selectors", compareSelectors(left, right, false));
            result.put("selectorsCi", compareSelectors(left, right, true));
            result.put("resultSelectors", resultSelectorRows(resultFile.toString()));
            result.put("resultSelectorsCi", resultSelectorSummary(resultFile.toString()));
            result.put("inspectfields", compareInspectFields(false));
            result.put("inspectfieldsCi", compareInspectFields(true));
            result.put("resultFields", resultInspectFields(false));
            result.put("resultFieldsCi", resultInspectFields(true));
            result.put("resultAliases", resultAliasRows(resultFile.toString()));
            result.put("resultAliasesCi", resultAliasSummary(resultFile.toString()));
            result.put("resultSummary", compareResultSummary(resultFile.toString(), result));
            result.put("resultSummaryLine", compareResultSummaryLine(asMap(result.get("resultSummary"))));
            result.put("resultDashboard", resultDashboard(resultFile.toString(), result));
            result.put("resultDashboardCi", resultDashboardLine(asMap(result.get("resultDashboard"))));
            result.put("resultDashboardManifest", resultDashboardManifest(resultFile.toString(), result));
            result.put("resultDashboardManifestCi", resultDashboardManifestLine(asMap(result.get("resultDashboardManifest"))));
            result.put("resultDashboardSummary", resultDashboardSummary(resultFile.toString(), result));
            result.put("resultDashboardSummaryCi", resultDashboardSummaryLine(asMap(result.get("resultDashboardSummary"))));
            result.put("resultDashboardOverview", resultDashboardOverview(resultFile.toString(), result));
            result.put("resultDashboardOverviewCi", resultDashboardOverviewLine(asMap(result.get("resultDashboardOverview"))));
            result.put("resultDashboardStatus", resultDashboardStatus(resultFile.toString(), result));
            result.put("resultDashboardStatusCi", resultDashboardStatusLine(asMap(result.get("resultDashboardStatus"))));
            result.put("resultDashboardHealth", resultDashboardHealth(resultFile.toString(), result));
            result.put("resultDashboardHealthCi", resultDashboardHealthLine(asMap(result.get("resultDashboardHealth"))));
            result.put("resultDashboardAliases", resultDashboardAliasRows());
            result.put("resultDashboardAliasesCi", resultDashboardAliasSummary());
            result.put("resultDashboardTree", resultDashboardTree(resultFile.toString(), result));
            result.put("resultDashboardTreeCi", resultDashboardTreeLine(asMap(result.get("resultDashboardTree"))));
            result.put("resultDashboardTreeManifest", resultDashboardTree(resultFile.toString(), result));
            result.put("resultDashboardTreeManifestCi", resultDashboardTreeLine(asMap(result.get("resultDashboardTreeManifest"))));
            result.put("resultDashboardTreeSummary", resultDashboardTree(resultFile.toString(), result));
            result.put("resultDashboardTreeSummaryCi", resultDashboardTreeLine(asMap(result.get("resultDashboardTreeSummary"))));
            result.put("resultDashboardTreeDashboard", resultDashboardTreeDashboard(resultFile.toString(), result));
            result.put("resultDashboardTreeDashboardCi", resultDashboardTreeDashboardLine(asMap(result.get("resultDashboardTreeDashboard"))));
            result.put("resultDashboardTreePaths", resultDashboardTreePathRows(resultFile.toString()));
            result.put("resultDashboardTreePathsCi", resultDashboardTreePathSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalog", resultDashboardTreeCatalog(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogCi", resultDashboardTreeCatalogLine(asMap(result.get("resultDashboardTreeCatalog"))));
            result.put("resultDashboardTreeCatalogManifest", resultDashboardTreeCatalog(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogManifestCi", resultDashboardTreeCatalogLine(asMap(result.get("resultDashboardTreeCatalogManifest"))));
            result.put("resultDashboardTreeCatalogSummary", resultDashboardTreeCatalog(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogSummaryCi", resultDashboardTreeCatalogLine(asMap(result.get("resultDashboardTreeCatalogSummary"))));
            result.put("resultDashboardTreeCatalogOverview", resultDashboardTreeCatalog(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogOverviewCi", resultDashboardTreeCatalogLine(asMap(result.get("resultDashboardTreeCatalogOverview"))));
            result.put("resultDashboardTreeCatalogTreeSummary", resultDashboardTreeCatalogTreeSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryCi", resultDashboardTreeCatalogTreeSummaryLine(asMap(result.get("resultDashboardTreeCatalogTreeSummary"))));
            result.put("resultDashboardTreeCatalogTreeSummaryFields", resultDashboardTreeCatalogTreeSummaryFieldRows());
            result.put("resultDashboardTreeCatalogTreeSummaryFieldsCi", resultDashboardTreeCatalogTreeSummaryFieldSummary());
            result.put("resultDashboardTreeCatalogTreeSummaryCommands", resultDashboardTreeCatalogTreeSummaryCommandRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryCommandsCi", resultDashboardTreeCatalogTreeSummaryCommandSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummarySelectors", resultDashboardTreeCatalogTreeSummarySelectorRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummarySelectorsCi", resultDashboardTreeCatalogTreeSummarySelectorSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryAliases", resultDashboardTreeCatalogTreeSummaryAliasRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryAliasesCi", resultDashboardTreeCatalogTreeSummaryAliasSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryDiscovery", resultDashboardTreeCatalogTreeSummaryDiscovery(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryDiscoveryCi", resultDashboardTreeCatalogTreeSummaryDiscoverySummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryLineage", resultDashboardTreeCatalogTreeSummaryLineageRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryLineageCi", resultDashboardTreeCatalogTreeSummaryLineageSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryNavigation", resultDashboardTreeCatalogTreeSummaryNavigation(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryNavigationCi", resultDashboardTreeCatalogTreeSummaryNavigationSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryBreadcrumbs", resultDashboardTreeCatalogTreeSummaryBreadcrumbs(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryBreadcrumbsCi", resultDashboardTreeCatalogTreeSummaryBreadcrumbs(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummarySitemap", resultDashboardTreeCatalogTreeSummarySitemapRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummarySitemapCi", resultDashboardTreeCatalogTreeSummarySitemapSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryIndex", resultDashboardTreeCatalogTreeSummaryIndex(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryIndexCi", resultDashboardTreeCatalogTreeSummaryIndexSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistry", resultDashboardTreeCatalogTreeSummaryRegistry(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryCi", resultDashboardTreeCatalogTreeSummaryRegistrySummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryFields", resultDashboardTreeCatalogTreeSummaryRegistryFieldRows());
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryFieldsCi", resultDashboardTreeCatalogTreeSummaryRegistryFieldSummary());
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryCommands", resultDashboardTreeCatalogTreeSummaryRegistryCommandRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryCommandsCi", resultDashboardTreeCatalogTreeSummaryRegistryCommandSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySelectors", resultDashboardTreeCatalogTreeSummaryRegistrySelectorRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySelectorsCi", resultDashboardTreeCatalogTreeSummaryRegistrySelectorSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryAliases", resultDashboardTreeCatalogTreeSummaryRegistryAliasRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryAliasesCi", resultDashboardTreeCatalogTreeSummaryRegistryAliasSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryDiscovery", resultDashboardTreeCatalogTreeSummaryRegistryDiscovery(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryCi", resultDashboardTreeCatalogTreeSummaryRegistryDiscoverySummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryNavigation", resultDashboardTreeCatalogTreeSummaryRegistryNavigation(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryNavigationCi", resultDashboardTreeCatalogTreeSummaryRegistryNavigationSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs", resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsCi", resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySitemap", resultDashboardTreeCatalogTreeSummaryRegistrySitemapRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySitemapCi", resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryIndex", resultDashboardTreeCatalogTreeSummaryRegistryIndex(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryIndexCi", resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryOverview", resultDashboardTreeCatalogTreeSummaryRegistryOverview(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryOverviewCi", resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryStatus", resultDashboardTreeCatalogTreeSummaryRegistryStatus(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryStatusCi", resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryHealth", resultDashboardTreeCatalogTreeSummaryRegistryHealth(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryHealthCi", resultDashboardTreeCatalogTreeSummaryRegistryHealthSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySummary", resultDashboardTreeCatalogTreeSummaryRegistrySummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySummaryCi", resultDashboardTreeCatalogTreeSummaryRegistrySummarySummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryManifest", resultDashboardTreeCatalogTreeSummaryRegistryManifest(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryManifestCi", resultDashboardTreeCatalogTreeSummaryRegistryManifestSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryEnvelope", resultDashboardTreeCatalogTreeSummaryRegistryEnvelope(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeCi", resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySnapshot", resultDashboardTreeCatalogTreeSummaryRegistrySnapshot(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistrySnapshotCi", resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryAudit", resultDashboardTreeCatalogTreeSummaryRegistryAudit(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryAuditCi", resultDashboardTreeCatalogTreeSummaryRegistryAuditSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryReport", resultDashboardTreeCatalogTreeSummaryRegistryReport(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryReportCi", resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklist", resultDashboardTreeCatalogTreeSummaryRegistryChecklist(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus", resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth", resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard", resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel", resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget", resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile", resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric", resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend", resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar", resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliveryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliverySummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistoryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistorySummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetrySummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundle(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnvCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdownCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadgeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanelCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidgetCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTileCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetricCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrendCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGaugeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBarCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenuCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbarCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPaletteCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndexCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResultsCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenterCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadgeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdownCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnvCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBarCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenuCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbarCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPaletteCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndexCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResultsCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastStateCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenterCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadgeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdownCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnvCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnvCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogFields", resultDashboardTreeCatalogFieldRows());
            result.put("resultDashboardTreeCatalogFieldsCi", resultDashboardTreeCatalogFieldSummary());
            result.put("resultDashboardTreeCatalogCommands", resultDashboardTreeCatalogCommandRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogCommandsCi", resultDashboardTreeCatalogCommandSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogCommandsSummary", resultDashboardTreeCatalogCommandSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogCommandsSummaryCi", resultDashboardTreeCatalogCommandSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogSelectors", resultDashboardTreeCatalogSelectorRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogSelectorsCi", resultDashboardTreeCatalogSelectorSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogSelectorsSummary", resultDashboardTreeCatalogSelectorSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogSelectorsSummaryCi", resultDashboardTreeCatalogSelectorSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogAliases", resultDashboardTreeCatalogAliasRows(resultFile.toString()));
            result.put("resultDashboardTreeCatalogAliasesCi", resultDashboardTreeCatalogAliasSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogAliasesSummary", resultDashboardTreeCatalogAliasSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogAliasesSummaryCi", resultDashboardTreeCatalogAliasSummary(resultFile.toString()));
            result.put("resultDashboardTreeCatalogChildrenSummary", resultDashboardTreeChildSummary());
            result.put("resultDashboardTreeCatalogChildrenSummaryCi", resultDashboardTreeChildSummary());
            result.put("resultDashboardTreeCatalogCountsSummary", resultDashboardTreeCountSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCatalogCountsSummaryCi", resultDashboardTreeCountLine(asMap(result.get("resultDashboardTreeCatalogCountsSummary"))));
            result.put("resultDashboardTreeChildren", resultDashboardTreeChildRows());
            result.put("resultDashboardTreeChildrenCi", resultDashboardTreeChildSummary());
            result.put("resultDashboardTreeCounts", resultDashboardTreeCountSummary(resultFile.toString(), result));
            result.put("resultDashboardTreeCountsCi", resultDashboardTreeCountLine(asMap(result.get("resultDashboardTreeCounts"))));
            result.put("resultDashboardTreeFields", resultDashboardTreeFieldRows());
            result.put("resultDashboardTreeFieldsCi", resultDashboardTreeFieldSummary());
            result.put("resultDashboardTreeAliases", resultDashboardTreeAliasRows());
            result.put("resultDashboardTreeAliasesCi", resultDashboardTreeAliasSummary());
            result.put("resultDashboardTreeCommands", resultDashboardTreeCommandRows(resultFile.toString()));
            result.put("resultDashboardTreeCommandsCi", resultDashboardTreeCommandSummary(resultFile.toString()));
            result.put("resultDashboardTreeSelectors", resultDashboardTreeSelectorRows(resultFile.toString()));
            result.put("resultDashboardTreeSelectorsCi", resultDashboardTreeSelectorSummary(resultFile.toString()));
            result.put("resultDashboardFamilyFields", resultDashboardFamilyFieldRows());
            result.put("resultDashboardFamilyFieldsCi", resultDashboardFamilyFieldSummary());
            result.put("resultDashboardFamilyCommands", resultDashboardFamilyCommandRows(resultFile.toString()));
            result.put("resultDashboardFamilyCommandsCi", resultDashboardFamilyCommandSummary(resultFile.toString()));
            result.put("resultDashboardFamilySelectors", resultDashboardFamilySelectorRows(resultFile.toString()));
            result.put("resultDashboardFamilySelectorsCi", resultDashboardFamilySelectorSummary(resultFile.toString()));
            result.put("resultDashboardSelectors", resultDashboardSelectorRows(resultFile.toString()));
            result.put("resultDashboardSelectorsCi", resultDashboardSelectorSummary(resultFile.toString()));
            result.put("resultDashboardCommands", resultDashboardCommandRows(resultFile.toString()));
            result.put("resultDashboardCommandsCi", resultDashboardCommandSummary(resultFile.toString()));
            result.put("resultDashboardFields", resultDashboardFieldRows());
            result.put("resultDashboardFieldsCi", resultDashboardFieldSummary());
            result.put("resultManifestCi", result.get("resultSummaryLine"));
            Map<String, Object> resultManifest = new LinkedHashMap<>(result);
            result.put("resultManifest", resultManifest);
            result.put("resultFile", resultFile.toString());
            JSON.writeValue(resultFile.toFile(), result);
            return result;
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to write compare result to " + String.valueOf(outputFile) + ".", exception);
        }
    }

    private static Map<String, Object> summarizeResultSummary(ResultInput result) {
        Object fromResult = result.result().get("resultSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return compareResultSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultSummaryLine(ResultInput result) {
        Object fromResult = result.result().get("resultSummaryLine");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return compareResultSummaryLine(summarizeResultSummary(result));
    }

    private static Map<String, Object> summarizeResultDashboard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboard");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardLine(summarizeResultDashboard(result));
    }

    private static Map<String, Object> summarizeResultDashboardManifest(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardManifest");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardManifest(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardManifestCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardManifestCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardManifestLine(summarizeResultDashboardManifest(result));
    }

    private static Map<String, Object> summarizeResultDashboardSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardSummaryCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardSummaryLine(summarizeResultDashboardSummary(result));
    }

    private static Map<String, Object> summarizeResultDashboardOverview(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardOverview");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardOverview(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardOverviewCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardOverviewCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardOverviewLine(summarizeResultDashboardOverview(result));
    }

    private static Map<String, Object> summarizeResultDashboardStatus(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardStatus");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardStatus(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardStatusCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardStatusCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardStatusLine(summarizeResultDashboardStatus(result));
    }

    private static Map<String, Object> summarizeResultDashboardHealth(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardHealth");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardHealth(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardHealthCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardHealthCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardHealthLine(summarizeResultDashboardHealth(result));
    }

    private static List<Map<String, Object>> summarizeResultDashboardFields(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFields");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardFieldRows();
    }

    private static Map<String, Object> summarizeResultDashboardFieldsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFieldsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardFieldSummary();
    }

    private static List<Map<String, Object>> summarizeResultDashboardCommands(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardCommands");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardCommandRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardCommandsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardCommandsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardAliases(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardAliases");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardAliasRows();
    }

    private static Map<String, Object> summarizeResultDashboardAliasesCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardAliasesCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardAliasSummary();
    }

    private static Map<String, Object> summarizeResultDashboardTree(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTree");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTree(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeLine(summarizeResultDashboardTree(result));
    }

    private static Map<String, Object> summarizeResultDashboardTreeManifest(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeManifest");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTree(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeManifestCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeManifestCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeLine(summarizeResultDashboardTreeManifest(result));
    }

    private static Map<String, Object> summarizeResultDashboardTreeSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTree(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeSummaryCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeLine(summarizeResultDashboardTreeSummary(result));
    }

    private static Map<String, Object> summarizeResultDashboardTreeDashboard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeDashboard");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeDashboard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeDashboardCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeDashboardCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeDashboardLine(summarizeResultDashboardTreeDashboard(result));
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreePaths(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreePaths");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreePathRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreePathsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreePathsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreePathSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalog(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalog");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalog(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeCatalogLine(summarizeResultDashboardTreeCatalog(result));
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogManifest(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogManifest");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return summarizeResultDashboardTreeCatalog(result);
    }

    private static String summarizeResultDashboardTreeCatalogManifestCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogManifestCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeCatalogLine(summarizeResultDashboardTreeCatalogManifest(result));
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return summarizeResultDashboardTreeCatalog(result);
    }

    private static String summarizeResultDashboardTreeCatalogSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogSummaryCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeCatalogLine(summarizeResultDashboardTreeCatalogSummary(result));
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogOverview(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogOverview");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return summarizeResultDashboardTreeCatalog(result);
    }

    private static String summarizeResultDashboardTreeCatalogOverviewCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogOverviewCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeCatalogLine(summarizeResultDashboardTreeCatalogOverview(result));
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeCatalogTreeSummaryLine(summarizeResultDashboardTreeCatalogTreeSummary(result));
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryFields(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryFields");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryFieldRows();
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryFieldsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryFieldsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryFieldSummary();
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryCommands(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryCommands");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryCommandRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryCommandsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryCommandsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummarySelectors(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummarySelectors");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummarySelectorRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummarySelectorsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummarySelectorsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummarySelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryAliases(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryAliases");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryAliasRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryAliasesCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryAliasesCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryAliasSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryDiscovery(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryDiscovery");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryDiscovery(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryDiscoveryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryDiscoveryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryDiscoverySummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryLineage(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryLineage");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryLineageRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryLineageCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryLineageCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryLineageSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryNavigation(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryNavigation");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryNavigation(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryNavigationCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryNavigationCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryNavigationSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryBreadcrumbs(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryBreadcrumbs");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryBreadcrumbs(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryBreadcrumbsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryBreadcrumbsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryBreadcrumbs(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummarySitemap(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummarySitemap");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummarySitemapRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummarySitemapCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummarySitemapCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummarySitemapSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryIndex(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryIndex");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryIndex(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryIndexCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryIndexCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryIndexSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistry(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistry");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistry(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryRegistryFields(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryFields");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryFieldRows();
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryFieldsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryFieldsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryFieldSummary();
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryRegistryCommands(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryCommands");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryCommandRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryCommandsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryCommandsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySelectors(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySelectors");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySelectorRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySelectorsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySelectorsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryRegistryAliases(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryAliases");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryAliasRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryAliasesCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryAliasesCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryAliasSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryDiscovery(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryDiscovery");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryDiscovery(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryDiscoveryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryDiscoverySummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryNavigation(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryNavigation");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryNavigation(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryNavigationCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryNavigationCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryNavigationSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySitemap(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySitemap");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySitemapRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySitemapCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySitemapCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryIndex(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryIndex");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryIndex(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryIndexCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryIndexCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryOverview(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryOverview");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryOverview(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryOverviewCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryOverviewCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryStatus(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryStatus");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryStatus(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryStatusCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryStatusCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryHealth(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryHealth");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryHealth(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryHealthCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryHealthCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryHealthSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySummaryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySummarySummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryManifest(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryManifest");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryManifest(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryManifestCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryManifestCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryManifestSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryEnvelope(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryEnvelope");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryEnvelope(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryEnvelopeCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySnapshot(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySnapshot");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySnapshot(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistrySnapshotCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistrySnapshotCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryAudit(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryAudit");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryAudit(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryAuditCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryAuditCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryAuditSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryReport(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryReport");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryReport(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryReportCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryReportCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklist(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklist");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklist(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistCard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistCardCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTile(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTileCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliveryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliveryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliverySummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistoryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistoryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistorySummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetrySummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine");
        if (fromResult instanceof String) {
            return (String) fromResult;
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineCi");
        if (fromResult instanceof String) {
            return (String) fromResult;
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv");
        if (fromResult instanceof String) {
            return (String) fromResult;
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvCi");
        if (fromResult instanceof String) {
            return (String) fromResult;
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown");
        if (fromResult instanceof String) {
            return (String) fromResult;
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownCi");
        if (fromResult instanceof String) {
            return (String) fromResult;
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundle(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptCi");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeCi");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnvCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdownCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadgeCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCardCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanelCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidgetCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTileCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetricCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrendCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGaugeCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCardCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBarCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenuCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbarCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPaletteCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndexCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResultsCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenterCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadgeCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdownCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnvCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCardCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBarCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenuCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbarCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPaletteCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndexCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResultsCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastStateCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenterCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge");
        if (fromResult instanceof Map) {
            return asMap((Map<?, ?>) fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadgeCi");
        if (fromResultCi instanceof Map) {
            return asMap((Map<?, ?>) fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdownCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnvCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnvCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine");
        if (fromResult instanceof String) {
            return String.valueOf(fromResult);
        }
        Object fromResultCi = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLineCi");
        if (fromResultCi instanceof String) {
            return String.valueOf(fromResultCi);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogFields(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogFields");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogFieldRows();
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogFieldsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogFieldsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogFieldSummary();
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogCommands(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogCommands");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogCommandRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogCommandsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogCommandsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogCommandsSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogCommandsSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogCommandsSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogCommandsSummaryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogSelectors(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogSelectors");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogSelectorRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogSelectorsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogSelectorsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogSelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogSelectorsSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogSelectorsSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogSelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogSelectorsSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogSelectorsSummaryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogSelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCatalogAliases(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogAliases");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCatalogAliasRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogAliasesCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogAliasesCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogAliasSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogAliasesSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogAliasesSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogAliasSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogAliasesSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogAliasesSummaryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCatalogAliasSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogChildrenSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogChildrenSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeChildSummary();
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogChildrenSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogChildrenSummaryCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeChildSummary();
    }

    private static Map<String, Object> summarizeResultDashboardTreeCatalogCountsSummary(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogCountsSummary");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCountSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCatalogCountsSummaryCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCatalogCountsSummaryCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeCountLine(summarizeResultDashboardTreeCatalogCountsSummary(result));
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeChildren(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeChildren");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeChildRows();
    }

    private static Map<String, Object> summarizeResultDashboardTreeChildrenCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeChildrenCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeChildSummary();
    }

    private static Map<String, Object> summarizeResultDashboardTreeCounts(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCounts");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCountSummary(result.resultFile().toAbsolutePath().normalize().toString(), result.result());
    }

    private static String summarizeResultDashboardTreeCountsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCountsCi");
        if (fromResult instanceof String) {
            String value = (String) fromResult;
            if (!value.isBlank()) {
                return value;
            }
        }
        return resultDashboardTreeCountLine(summarizeResultDashboardTreeCounts(result));
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeFields(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeFields");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeFieldRows();
    }

    private static Map<String, Object> summarizeResultDashboardTreeFieldsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeFieldsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeFieldSummary();
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeAliases(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeAliases");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeAliasRows();
    }

    private static Map<String, Object> summarizeResultDashboardTreeAliasesCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeAliasesCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeAliasSummary();
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeCommands(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCommands");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeCommandRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeCommandsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeCommandsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardTreeSelectors(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeSelectors");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardTreeSelectorRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardTreeSelectorsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardTreeSelectorsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardTreeSelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardFamilyFields(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFamilyFields");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardFamilyFieldRows();
    }

    private static Map<String, Object> summarizeResultDashboardFamilyFieldsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFamilyFieldsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardFamilyFieldSummary();
    }

    private static List<Map<String, Object>> summarizeResultDashboardFamilyCommands(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFamilyCommands");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardFamilyCommandRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardFamilyCommandsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFamilyCommandsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardFamilyCommandSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardFamilySelectors(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFamilySelectors");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardFamilySelectorRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardFamilySelectorsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardFamilySelectorsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardFamilySelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static List<Map<String, Object>> summarizeResultDashboardSelectors(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardSelectors");
        if (fromResult instanceof List) {
            List<?> list = (List) fromResult;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map = (Map) item;
                    rows.add(asMap(map));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultDashboardSelectorRows(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Map<String, Object> summarizeResultDashboardSelectorsCi(ResultInput result) {
        Object fromResult = result.result().get("resultDashboardSelectorsCi");
        if (fromResult instanceof Map) {
            Map<?, ?> map = (Map) fromResult;
            return asMap(map);
        }
        return resultDashboardSelectorSummary(result.resultFile().toAbsolutePath().normalize().toString());
    }

    private static Object summarizeResultCommands(ResultInput result, boolean ci) {
        String resultFile = result.resultFile().toAbsolutePath().normalize().toString();
        if (ci) {
            Object fromResult = result.result().get("resultCommandsCi");
            if (fromResult instanceof Map) {
                Map<?, ?> map = (Map) fromResult;
                return asMap(map);
            }
            return resultCommandSummary(resultFile);
        }
        Object fromResult2 = result.result().get("resultCommands");
        if (fromResult2 instanceof List) {
            List<?> list = (List) fromResult2;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map2 = (Map) item;
                    rows.add(asMap(map2));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultCommandRows(resultFile);
    }

    private static Object summarizeResultSelectors(ResultInput result, boolean ci) {
        String resultFile = result.resultFile().toAbsolutePath().normalize().toString();
        if (ci) {
            Object fromResult = result.result().get("resultSelectorsCi");
            if (fromResult instanceof Map) {
                Map<?, ?> map = (Map) fromResult;
                return asMap(map);
            }
            return resultSelectorSummary(resultFile);
        }
        Object fromResult2 = result.result().get("resultSelectors");
        if (fromResult2 instanceof List) {
            List<?> list = (List) fromResult2;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map2 = (Map) item;
                    rows.add(asMap(map2));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultSelectorRows(resultFile);
    }

    private static Object summarizeResultFields(ResultInput result, boolean ci) {
        if (ci) {
            Object fromResult = result.result().get("resultFieldsCi");
            if (fromResult instanceof Map) {
                Map<?, ?> map = (Map) fromResult;
                return asMap(map);
            }
            return resultInspectFields(true);
        }
        Object fromResult2 = result.result().get("resultFields");
        if (fromResult2 instanceof List) {
            List<?> list = (List) fromResult2;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map2 = (Map) item;
                    rows.add(asMap(map2));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultInspectFields(false);
    }

    private static Map<String, Object> summarizeResultPayloads(ResultInput result) {
        Map<String, Object> payloads = new LinkedHashMap<>();
        payloads.put("summary", "payloads");
        payloads.put("reportVersion", result.result().getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        payloads.put("resultFile", result.resultFile().toAbsolutePath().normalize().toString());
        payloads.put("leftReportFile", result.result().getOrDefault("leftReportFile", ""));
        payloads.put("rightReportFile", result.result().getOrDefault("rightReportFile", ""));
        payloads.put("leftReportVersion", result.result().getOrDefault("leftReportVersion", ""));
        payloads.put("rightReportVersion", result.result().getOrDefault("rightReportVersion", ""));
        return payloads;
    }

    private static String summarizeResultPayloadsLine(ResultInput result) {
        Map<String, Object> payloads = summarizeResultPayloads(result);
        return "resultPayloads reportVersion=" + String.valueOf(payloads.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(payloads.getOrDefault("resultFile", "")) + " leftReportFile=" + String.valueOf(payloads.getOrDefault("leftReportFile", "")) + " rightReportFile=" + String.valueOf(payloads.getOrDefault("rightReportFile", "")) + " leftReportVersion=" + String.valueOf(payloads.getOrDefault("leftReportVersion", "")) + " rightReportVersion=" + String.valueOf(payloads.getOrDefault("rightReportVersion", ""));
    }

    private static Object summarizeResultAliases(ResultInput result, boolean ci) {
        String resultFile = result.resultFile().toAbsolutePath().normalize().toString();
        if (ci) {
            Object fromResult = result.result().get("resultAliasesCi");
            if (fromResult instanceof Map) {
                Map<?, ?> map = (Map) fromResult;
                return asMap(map);
            }
            return resultAliasSummary(resultFile);
        }
        Object fromResult2 = result.result().get("resultAliases");
        if (fromResult2 instanceof List) {
            List<?> list = (List) fromResult2;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map2 = (Map) item;
                    rows.add(asMap(map2));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return resultAliasRows(resultFile);
    }

    private static Map<String, Object> resultAliasRow(String name, String selector, String resultFile) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", name);
        row.put("selector", selector);
        row.put("command", "jbang trainer/trainer_byte_latent_train_infer_compare_inspector.java " + resultFile + " " + selector);
        return row;
    }

    private static Map<String, Object> resultCheckRow(String name, boolean passed, String description) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", name);
        row.put("passed", Boolean.valueOf(passed));
        row.put("description", description);
        return row;
    }

    private static List<Map<String, Object>> resultAliasRows(String resultFile) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(resultAliasRow("overview", "resultOverview:ci", resultFile));
        rows.add(resultAliasRow("manifest", "resultManifest:ci", resultFile));
        rows.add(resultAliasRow("comparison", "resultComparison:ci", resultFile));
        rows.add(resultAliasRow("sources", "resultSources:ci", resultFile));
        rows.add(resultAliasRow("status", "resultStatus:ci", resultFile));
        rows.add(resultAliasRow("health", "resultHealth:ci", resultFile));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultAliasSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultAliasRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "short", "resultFile", resultFile, "aliasCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static List<Map<String, Object>> resultCommandRows(String resultFile) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(resultAliasRow("overview", "resultOverview:ci", resultFile));
        rows.add(resultAliasRow("manifest", "resultManifest:ci", resultFile));
        rows.add(resultAliasRow("comparison", "resultComparison:ci", resultFile));
        rows.add(resultAliasRow("sources", "resultSources:ci", resultFile));
        rows.add(resultAliasRow("aliases", "resultAliases:ci", resultFile));
        rows.add(resultAliasRow("status", "resultStatus:ci", resultFile));
        rows.add(resultAliasRow("health", "resultHealth:ci", resultFile));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultCommandSummary(String resultFile) {
        List<String> commands = new ArrayList<>();
        for (Map<String, Object> row : resultCommandRows(resultFile)) {
            commands.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "short", "resultFile", resultFile, "commandCount", Integer.valueOf(commands.size()), "commands", commands);
    }

    private static List<Map<String, Object>> resultSelectorRows(String resultFile) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(resultAliasRow("commands", "resultCommands:ci", resultFile));
        rows.add(resultAliasRow("overview", "resultOverview:ci", resultFile));
        rows.add(resultAliasRow("manifest", "resultManifest:ci", resultFile));
        rows.add(resultAliasRow("comparison", "resultComparison:ci", resultFile));
        rows.add(resultAliasRow("sources", "resultSources:ci", resultFile));
        rows.add(resultAliasRow("aliases", "resultAliases:ci", resultFile));
        rows.add(resultAliasRow("status", "resultStatus:ci", resultFile));
        rows.add(resultAliasRow("health", "resultHealth:ci", resultFile));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultSelectorSummary(String resultFile) {
        List<String> commands = new ArrayList<>();
        for (Map<String, Object> row : resultSelectorRows(resultFile)) {
            commands.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "short", "resultFile", resultFile, "selectorCount", Integer.valueOf(commands.size()), "selectors", commands);
    }

    private static Object resultInspectFields(boolean ci) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(compareInspectFieldRow("reportVersion", "string", "Standalone compare-result schema marker"));
        rows.add(compareInspectFieldRow("resultFile", "string", "Standalone compare-result file path"));
        rows.add(compareInspectFieldRow("leftReportVersion", "string", "Left train-infer report schema marker"));
        rows.add(compareInspectFieldRow("rightReportVersion", "string", "Right train-infer report schema marker"));
        rows.add(compareInspectFieldRow("leftReportFile", "string", "Left train-infer report path"));
        rows.add(compareInspectFieldRow("rightReportFile", "string", "Right train-infer report path"));
        rows.add(compareInspectFieldRow("resultSummary", "object", "Compact reopened-result summary block"));
        rows.add(compareInspectFieldRow("resultSummaryLine", "string", "Human-readable one-line reopened-result summary"));
        rows.add(compareInspectFieldRow("resultCommands", "array", "Alias-oriented reopened-result command rows"));
        rows.add(compareInspectFieldRow("resultCommandsCi", "object", "Compact reopened-result command summary"));
        rows.add(compareInspectFieldRow("resultSelectors", "array", "Alias-oriented reopened-result selector rows"));
        rows.add(compareInspectFieldRow("resultSelectorsCi", "object", "Compact reopened-result selector summary"));
        rows.add(compareInspectFieldRow("resultFields", "array", "Standalone compare-result field discovery rows"));
        rows.add(compareInspectFieldRow("resultFieldsCi", "object", "Compact standalone compare-result field summary"));
        rows.add(compareInspectFieldRow("resultOverview", "object", "Artifact-centric reopened-result overview alias"));
        rows.add(compareInspectFieldRow("resultManifest", "object", "Artifact-centric reopened-result full payload alias"));
        rows.add(compareInspectFieldRow("resultComparison", "object", "Artifact-centric reopened-result comparison alias"));
        rows.add(compareInspectFieldRow("resultPayloads", "object", "Artifact-centric reopened-result payload references"));
        rows.add(compareInspectFieldRow("resultSources", "object", "Artifact-centric reopened-result source references"));
        rows.add(compareInspectFieldRow("resultAliases", "array", "Artifact-centric reopened-result alias discovery rows"));
        rows.add(compareInspectFieldRow("resultStatus", "object", "Artifact-centric reopened-result status alias"));
        rows.add(compareInspectFieldRow("resultHealth", "object", "Artifact-centric reopened-result health alias"));
        if (ci) {
            return Map.of("summary", "short", "fieldCount", Integer.valueOf(rows.size()), "fields", rows.stream().map(row -> {
                return row.get("name");
            }).toList());
        }
        return List.copyOf(rows);
    }

    private static Map<String, Object> compareResultSummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "short");
        summary.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        summary.put("resultFile", resultFile);
        summary.put("leftReportFile", result.getOrDefault("leftReportFile", ""));
        summary.put("rightReportFile", result.getOrDefault("rightReportFile", ""));
        summary.put("artifactSummaryLine", result.getOrDefault("artifactSummaryLine", ""));
        summary.put("compareSummaryLine", result.getOrDefault("compareSummaryLine", ""));
        return summary;
    }

    private static String compareResultSummaryLine(Map<String, Object> summary) {
        return "resultSummary reportVersion=" + String.valueOf(summary.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(summary.getOrDefault("resultFile", "")) + " leftReportFile=" + String.valueOf(summary.getOrDefault("leftReportFile", "")) + " rightReportFile=" + String.valueOf(summary.getOrDefault("rightReportFile", "")) + " compareSummaryLine=" + String.valueOf(summary.getOrDefault("compareSummaryLine", ""));
    }

    private static Map<String, Object> resultDashboard(String resultFile, Map<String, Object> result) {
        Map<String, Object> dashboard = new LinkedHashMap<>();
        dashboard.put("summary", "dashboard");
        dashboard.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        dashboard.put("resultFile", resultFile);
        dashboard.put("resultOverviewCi", result.getOrDefault("resultOverviewCi", ""));
        dashboard.put("resultComparisonCi", result.getOrDefault("resultComparisonCi", ""));
        dashboard.put("resultSourcesCi", summarizeResultPayloadsLine(new ResultInput(Path.of(resultFile, new String[0]), result)));
        dashboard.put("resultStatusCi", result.getOrDefault("resultStatusCi", ""));
        dashboard.put("resultHealthCi", result.getOrDefault("resultHealthCi", ""));
        dashboard.put("resultAliasesCi", result.getOrDefault("resultAliasesCi", Map.of()));
        dashboard.put("resultSelectorsCi", result.getOrDefault("resultDashboardSelectorsCi", resultDashboardSelectorSummary(resultFile)));
        return dashboard;
    }

    private static Map<String, Object> resultDashboardManifest(String resultFile, Map<String, Object> result) {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("summary", "dashboardManifest");
        manifest.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        manifest.put("resultFile", resultFile);
        manifest.put("resultDashboardCi", result.getOrDefault("resultDashboardCi", resultDashboardLine(resultDashboard(resultFile, result))));
        manifest.put("resultDashboardSummaryCi", result.getOrDefault("resultDashboardSummaryCi", resultDashboardSummaryLine(resultDashboardSummary(resultFile, result))));
        manifest.put("resultDashboardOverviewCi", result.getOrDefault("resultDashboardOverviewCi", resultDashboardOverviewLine(resultDashboardOverview(resultFile, result))));
        manifest.put("resultDashboardStatusCi", result.getOrDefault("resultDashboardStatusCi", resultDashboardStatusLine(resultDashboardStatus(resultFile, result))));
        manifest.put("resultDashboardHealthCi", result.getOrDefault("resultDashboardHealthCi", resultDashboardHealthLine(resultDashboardHealth(resultFile, result))));
        manifest.put("resultDashboardAliasesCi", result.getOrDefault("resultDashboardAliasesCi", resultDashboardAliasSummary()));
        manifest.put("resultDashboardSelectorsCi", result.getOrDefault("resultDashboardSelectorsCi", resultDashboardSelectorSummary(resultFile)));
        return manifest;
    }

    private static String resultDashboardManifestLine(Map<String, Object> manifest) {
        return "resultDashboardManifest reportVersion=" + String.valueOf(manifest.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(manifest.getOrDefault("resultFile", "")) + " resultDashboardSummaryCi=" + String.valueOf(manifest.getOrDefault("resultDashboardSummaryCi", "")) + " resultDashboardOverviewCi=" + String.valueOf(manifest.getOrDefault("resultDashboardOverviewCi", "")) + " resultDashboardStatusCi=" + String.valueOf(manifest.getOrDefault("resultDashboardStatusCi", "")) + " resultDashboardHealthCi=" + String.valueOf(manifest.getOrDefault("resultDashboardHealthCi", ""));
    }

    private static Map<String, Object> resultDashboardTree(String resultFile, Map<String, Object> result) {
        Map<String, Object> tree = new LinkedHashMap<>();
        tree.put("summary", "dashboardTree");
        tree.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        tree.put("resultFile", resultFile);
        tree.put("dashboard", Map.of("manifest", result.getOrDefault("resultDashboardManifestCi", resultDashboardManifestLine(resultDashboardManifest(resultFile, result))), "summary", result.getOrDefault("resultDashboardSummaryCi", resultDashboardSummaryLine(resultDashboardSummary(resultFile, result))), "overview", result.getOrDefault("resultDashboardOverviewCi", resultDashboardOverviewLine(resultDashboardOverview(resultFile, result))), "status", result.getOrDefault("resultDashboardStatusCi", resultDashboardStatusLine(resultDashboardStatus(resultFile, result))), "health", result.getOrDefault("resultDashboardHealthCi", resultDashboardHealthLine(resultDashboardHealth(resultFile, result)))));
        tree.put("aliases", result.getOrDefault("resultDashboardAliasesCi", resultDashboardAliasSummary()));
        tree.put("fields", result.getOrDefault("resultDashboardFamilyFieldsCi", resultDashboardFamilyFieldSummary()));
        tree.put("commands", result.getOrDefault("resultDashboardFamilyCommandsCi", resultDashboardFamilyCommandSummary(resultFile)));
        tree.put("selectors", result.getOrDefault("resultDashboardFamilySelectorsCi", resultDashboardFamilySelectorSummary(resultFile)));
        return tree;
    }

    private static String resultDashboardTreeLine(Map<String, Object> tree) {
        return "resultDashboardTree reportVersion=" + String.valueOf(tree.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(tree.getOrDefault("resultFile", "")) + " aliases=" + String.valueOf(asMap(tree.get("aliases")).getOrDefault("aliasCount", 0)) + " fields=" + String.valueOf(asMap(tree.get("fields")).getOrDefault("fieldCount", 0)) + " commands=" + String.valueOf(asMap(tree.get("commands")).getOrDefault("commandCount", 0)) + " selectors=" + String.valueOf(asMap(tree.get("selectors")).getOrDefault("selectorCount", 0));
    }

    private static Map<String, Object> resultDashboardTreeDashboard(String resultFile, Map<String, Object> result) {
        Map<String, Object> dashboard = new LinkedHashMap<>();
        dashboard.put("summary", "dashboardTreeDashboard");
        dashboard.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        dashboard.put("resultFile", resultFile);
        dashboard.put("manifest", result.getOrDefault("resultDashboardManifestCi", resultDashboardManifestLine(resultDashboardManifest(resultFile, result))));
        dashboard.put("summaryLine", result.getOrDefault("resultDashboardSummaryCi", resultDashboardSummaryLine(resultDashboardSummary(resultFile, result))));
        dashboard.put("overview", result.getOrDefault("resultDashboardOverviewCi", resultDashboardOverviewLine(resultDashboardOverview(resultFile, result))));
        dashboard.put("status", result.getOrDefault("resultDashboardStatusCi", resultDashboardStatusLine(resultDashboardStatus(resultFile, result))));
        dashboard.put("health", result.getOrDefault("resultDashboardHealthCi", resultDashboardHealthLine(resultDashboardHealth(resultFile, result))));
        return dashboard;
    }

    private static String resultDashboardTreeDashboardLine(Map<String, Object> dashboard) {
        return "resultDashboardTreeDashboard reportVersion=" + String.valueOf(dashboard.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(dashboard.getOrDefault("resultFile", "")) + " overview=" + String.valueOf(dashboard.getOrDefault("overview", "")) + " status=" + String.valueOf(dashboard.getOrDefault("status", "")) + " health=" + String.valueOf(dashboard.getOrDefault("health", ""));
    }

    private static List<Map<String, Object>> resultDashboardTreePathRows(String resultFile) {
        return List.of(resultAliasRow("tree", "resultDashboardTree:ci", resultFile), resultAliasRow("treeManifest", "resultDashboardTreeManifest:ci", resultFile), resultAliasRow("treeSummary", "resultDashboardTreeSummary:ci", resultFile), resultAliasRow("treeDashboard", "resultDashboardTreeDashboard:ci", resultFile), resultAliasRow("treeChildren", "resultDashboardTreeChildren:ci", resultFile), resultAliasRow("treeCounts", "resultDashboardTreeCounts:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreePathSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreePathRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "short", "pathCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static Map<String, Object> resultDashboardTreeCatalog(String resultFile, Map<String, Object> result) {
        Map<String, Object> catalog = new LinkedHashMap<>();
        catalog.put("summary", "resultDashboardTreeCatalog");
        catalog.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        catalog.put("resultFile", resultFile);
        catalog.put("paths", result.getOrDefault("resultDashboardTreePathsCi", resultDashboardTreePathSummary(resultFile)));
        catalog.put("children", result.getOrDefault("resultDashboardTreeChildrenCi", resultDashboardTreeChildSummary()));
        catalog.put("counts", result.getOrDefault("resultDashboardTreeCounts", resultDashboardTreeCountSummary(resultFile, result)));
        return catalog;
    }

    private static String resultDashboardTreeCatalogLine(Map<String, Object> catalog) {
        return "resultDashboardTreeCatalog reportVersion=" + String.valueOf(catalog.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(catalog.getOrDefault("resultFile", "")) + " pathCount=" + String.valueOf(asMap(catalog.get("paths")).getOrDefault("pathCount", 0)) + " childCount=" + String.valueOf(asMap(catalog.get("children")).getOrDefault("childCount", 0)) + " aliasCount=" + String.valueOf(asMap(catalog.get("counts")).getOrDefault("aliasCount", 0)) + " fieldCount=" + String.valueOf(asMap(catalog.get("counts")).getOrDefault("fieldCount", 0)) + " commandCount=" + String.valueOf(asMap(catalog.get("counts")).getOrDefault("commandCount", 0)) + " selectorCount=" + String.valueOf(asMap(catalog.get("counts")).getOrDefault("selectorCount", 0));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "resultDashboardTreeCatalogTreeSummary");
        summary.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        summary.put("resultFile", resultFile);
        summary.put("paths", result.getOrDefault("resultDashboardTreePathsCi", resultDashboardTreePathSummary(resultFile)));
        summary.put("children", result.getOrDefault("resultDashboardTreeCatalogChildrenSummary", resultDashboardTreeChildSummary()));
        summary.put("counts", result.getOrDefault("resultDashboardTreeCatalogCountsSummary", resultDashboardTreeCountSummary(resultFile, result)));
        return summary;
    }

    private static String resultDashboardTreeCatalogTreeSummaryLine(Map<String, Object> summary) {
        return "resultDashboardTreeCatalogTreeSummary reportVersion=" + String.valueOf(summary.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(summary.getOrDefault("resultFile", "")) + " pathCount=" + String.valueOf(asMap(summary.get("paths")).getOrDefault("pathCount", 0)) + " childCount=" + String.valueOf(asMap(summary.get("children")).getOrDefault("childCount", 0)) + " aliasCount=" + String.valueOf(asMap(summary.get("counts")).getOrDefault("aliasCount", 0)) + " fieldCount=" + String.valueOf(asMap(summary.get("counts")).getOrDefault("fieldCount", 0)) + " commandCount=" + String.valueOf(asMap(summary.get("counts")).getOrDefault("commandCount", 0)) + " selectorCount=" + String.valueOf(asMap(summary.get("counts")).getOrDefault("selectorCount", 0));
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryFieldRows() {
        return List.of(compareInspectFieldRow("paths", "object", "Compact path summary embedded in the catalog tree summary."), compareInspectFieldRow("children", "object", "Compact child summary embedded in the catalog tree summary."), compareInspectFieldRow("counts", "object", "Compact count summary embedded in the catalog tree summary."));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryFieldSummary() {
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryFields", "fieldCount", 3, "fields", List.of("paths", "children", "counts"));
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryCommandRows(String resultFile) {
        return List.of(resultAliasRow("summary", "resultDashboardTreeCatalogTreeSummary:ci", resultFile), resultAliasRow("fields", "resultDashboardTreeCatalogTreeSummaryFields:ci", resultFile), resultAliasRow("paths", "resultDashboardTreePaths:ci", resultFile), resultAliasRow("children", "resultDashboardTreeCatalogChildrenSummary:ci", resultFile), resultAliasRow("counts", "resultDashboardTreeCatalogCountsSummary:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryCommandSummary(String resultFile) {
        List<String> commands = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummaryCommandRows(resultFile)) {
            commands.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryCommands", "commandCount", Integer.valueOf(commands.size()), "commands", commands);
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummarySelectorRows(String resultFile) {
        return List.of(resultAliasRow("summary", "resultDashboardTreeCatalogTreeSummary:ci", resultFile), resultAliasRow("fields", "resultDashboardTreeCatalogTreeSummaryFields:ci", resultFile), resultAliasRow("paths", "resultDashboardTreePaths:ci", resultFile), resultAliasRow("children", "resultDashboardTreeCatalogChildrenSummary:ci", resultFile), resultAliasRow("counts", "resultDashboardTreeCatalogCountsSummary:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummarySelectorSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummarySelectorRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummarySelectors", "selectorCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryAliasRows(String resultFile) {
        return List.of(resultAliasRow("summary", "resultDashboardTreeCatalogTreeSummary:ci", resultFile), resultAliasRow("fields", "resultDashboardTreeCatalogTreeSummaryFields:ci", resultFile), resultAliasRow("paths", "resultDashboardTreePaths:ci", resultFile), resultAliasRow("children", "resultDashboardTreeCatalogChildrenSummary:ci", resultFile), resultAliasRow("counts", "resultDashboardTreeCatalogCountsSummary:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryAliasSummary(String resultFile) {
        List<String> aliases = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummaryAliasRows(resultFile)) {
            aliases.add(String.valueOf(row.getOrDefault("name", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryAliases", "aliasCount", Integer.valueOf(aliases.size()), "aliases", aliases);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryDiscovery(String resultFile, Map<String, Object> result) {
        Map<String, Object> discovery = new LinkedHashMap<>();
        discovery.put("summary", "resultDashboardTreeCatalogTreeSummaryDiscovery");
        discovery.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        discovery.put("resultFile", resultFile);
        discovery.put("fields", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryFieldsCi", resultDashboardTreeCatalogTreeSummaryFieldSummary()));
        discovery.put("commands", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryCommandsCi", resultDashboardTreeCatalogTreeSummaryCommandSummary(resultFile)));
        discovery.put("selectors", result.getOrDefault("resultDashboardTreeCatalogTreeSummarySelectorsCi", resultDashboardTreeCatalogTreeSummarySelectorSummary(resultFile)));
        discovery.put("aliases", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryAliasesCi", resultDashboardTreeCatalogTreeSummaryAliasSummary(resultFile)));
        discovery.put("lineage", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryLineageCi", resultDashboardTreeCatalogTreeSummaryLineageSummary(resultFile)));
        return discovery;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryDiscoverySummary(String resultFile) {
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryDiscovery", "fieldCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryFieldRows().size()), "commandCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryCommandRows(resultFile).size()), "selectorCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummarySelectorRows(resultFile).size()), "aliasCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryAliasRows(resultFile).size()), "lineageCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryLineageRows(resultFile).size()), "sections", List.of("resultDashboardTreeCatalogTreeSummaryFields:ci", "resultDashboardTreeCatalogTreeSummaryCommands:ci", "resultDashboardTreeCatalogTreeSummarySelectors:ci", "resultDashboardTreeCatalogTreeSummaryAliases:ci", "resultDashboardTreeCatalogTreeSummaryLineage:ci"));
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryLineageRows(String resultFile) {
        return List.of(resultAliasRow("result", "resultManifest:ci", resultFile), resultAliasRow("dashboard", "resultDashboard:ci", resultFile), resultAliasRow("tree", "resultDashboardTree:ci", resultFile), resultAliasRow("catalog", "resultDashboardTreeCatalog:ci", resultFile), resultAliasRow("treeSummary", "resultDashboardTreeCatalogTreeSummary:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryLineageSummary(String resultFile) {
        List<String> lineage = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummaryLineageRows(resultFile)) {
            lineage.add(String.valueOf(row.getOrDefault("name", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryLineage", "lineageCount", Integer.valueOf(lineage.size()), "lineage", lineage);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryNavigation(String resultFile, Map<String, Object> result) {
        Map<String, Object> navigation = new LinkedHashMap<>();
        navigation.put("summary", "resultDashboardTreeCatalogTreeSummaryNavigation");
        navigation.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        navigation.put("resultFile", resultFile);
        navigation.put("current", "resultDashboardTreeCatalogTreeSummary:ci");
        navigation.put("lineage", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryLineageCi", resultDashboardTreeCatalogTreeSummaryLineageSummary(resultFile)));
        navigation.put("discovery", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryDiscoveryCi", resultDashboardTreeCatalogTreeSummaryDiscoverySummary(resultFile)));
        return navigation;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryNavigationSummary(String resultFile) {
        int size;
        Map<String, Object> lineage = resultDashboardTreeCatalogTreeSummaryLineageSummary(resultFile);
        Map<String, Object> discovery = resultDashboardTreeCatalogTreeSummaryDiscoverySummary(resultFile);
        Object lineageValues = lineage.getOrDefault("lineage", List.of());
        Object sectionValues = discovery.getOrDefault("sections", List.of());
        if (sectionValues instanceof List) {
            List<?> list = (List) sectionValues;
            size = list.size();
        } else {
            size = 0;
        }
        int sectionCount = size;
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryNavigation", "current", "resultDashboardTreeCatalogTreeSummary:ci", "lineageCount", lineage.getOrDefault("lineageCount", 0), "sectionCount", Integer.valueOf(sectionCount), "lineage", lineageValues, "sections", sectionValues);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryBreadcrumbs(String resultFile) {
        Map<String, Object> lineage = resultDashboardTreeCatalogTreeSummaryLineageSummary(resultFile);
        Object lineageValues = lineage.getOrDefault("lineage", List.of());
        List<String> parts = new ArrayList<>();
        if (lineageValues instanceof List) {
            List<?> list = (List) lineageValues;
            for (Object item : list) {
                parts.add(String.valueOf(item));
            }
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryBreadcrumbs", "current", "resultDashboardTreeCatalogTreeSummary:ci", "depth", Integer.valueOf(parts.size()), "breadcrumb", String.join("/", parts));
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummarySitemapRows(String resultFile) {
        return List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{resultAliasRow("summary", "resultDashboardTreeCatalogTreeSummary:ci", resultFile), resultAliasRow("fields", "resultDashboardTreeCatalogTreeSummaryFields:ci", resultFile), resultAliasRow("commands", "resultDashboardTreeCatalogTreeSummaryCommands:ci", resultFile), resultAliasRow("selectors", "resultDashboardTreeCatalogTreeSummarySelectors:ci", resultFile), resultAliasRow("aliases", "resultDashboardTreeCatalogTreeSummaryAliases:ci", resultFile), resultAliasRow("discovery", "resultDashboardTreeCatalogTreeSummaryDiscovery:ci", resultFile), resultAliasRow("lineage", "resultDashboardTreeCatalogTreeSummaryLineage:ci", resultFile), resultAliasRow("navigation", "resultDashboardTreeCatalogTreeSummaryNavigation:ci", resultFile), resultAliasRow("breadcrumbs", "resultDashboardTreeCatalogTreeSummaryBreadcrumbs:ci", resultFile), resultAliasRow("paths", "resultDashboardTreePaths:ci", resultFile), resultAliasRow("children", "resultDashboardTreeCatalogChildrenSummary:ci", resultFile), resultAliasRow("counts", "resultDashboardTreeCatalogCountsSummary:ci", resultFile)});
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummarySitemapSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummarySitemapRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummarySitemap", "endpointCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static List<String> resultDashboardTreeCatalogTreeSummarySelfIndex() {
        return List.of("resultDashboardTreeCatalogTreeSummary:ci");
    }

    private static List<String> resultDashboardTreeCatalogTreeSummaryDiscoveryIndex() {
        return List.of("resultDashboardTreeCatalogTreeSummaryFields:ci", "resultDashboardTreeCatalogTreeSummaryCommands:ci", "resultDashboardTreeCatalogTreeSummarySelectors:ci", "resultDashboardTreeCatalogTreeSummaryAliases:ci", "resultDashboardTreeCatalogTreeSummaryDiscovery:ci", "resultDashboardTreeCatalogTreeSummaryLineage:ci", "resultDashboardTreeCatalogTreeSummaryNavigation:ci", "resultDashboardTreeCatalogTreeSummaryBreadcrumbs:ci", "resultDashboardTreeCatalogTreeSummarySitemap:ci");
    }

    private static List<String> resultDashboardTreeCatalogTreeSummaryLeafIndex() {
        return List.of("resultDashboardTreePaths:ci", "resultDashboardTreeCatalogChildrenSummary:ci", "resultDashboardTreeCatalogCountsSummary:ci");
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryIndex(String resultFile) {
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryIndex", "resultFile", resultFile, "self", resultDashboardTreeCatalogTreeSummarySelfIndex(), "discovery", resultDashboardTreeCatalogTreeSummaryDiscoveryIndex(), "leaves", resultDashboardTreeCatalogTreeSummaryLeafIndex());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryIndexSummary(String resultFile) {
        List<String> self = resultDashboardTreeCatalogTreeSummarySelfIndex();
        List<String> discovery = resultDashboardTreeCatalogTreeSummaryDiscoveryIndex();
        List<String> leaves = resultDashboardTreeCatalogTreeSummaryLeafIndex();
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryIndex", "groupCount", 3, "endpointCount", Integer.valueOf(self.size() + discovery.size() + leaves.size()), "groups", List.of("self", "discovery", "leaves"), "selfCount", Integer.valueOf(self.size()), "discoveryCount", Integer.valueOf(discovery.size()), "leafCount", Integer.valueOf(leaves.size()));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistry(String resultFile) {
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistry", "resultFile", resultFile, "index", resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(resultFile), "sitemap", resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(resultFile), "breadcrumbs", resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistrySummary(String resultFile) {
        Map<String, Object> index = resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(resultFile);
        Map<String, Object> sitemap = resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(resultFile);
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistry", "sectionCount", 3, "sections", List.of("index", "sitemap", "breadcrumbs"), "groupCount", index.getOrDefault("groupCount", 0), "groupedEndpointCount", index.getOrDefault("endpointCount", 0), "flatEndpointCount", sitemap.getOrDefault("endpointCount", 0));
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryRegistryFieldRows() {
        return List.of(compareInspectFieldRow("index", "object", "Grouped endpoint index with self, metadata, and discovery counts."), compareInspectFieldRow("sitemap", "object", "Flat endpoint sitemap count and selectors."), compareInspectFieldRow("breadcrumbs", "object", "Log-friendly lineage breadcrumb path for the registry family."));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryFieldSummary() {
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryFields", "fieldCount", 3, "fields", List.of("index", "sitemap", "breadcrumbs"));
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryRegistryCommandRows(String resultFile) {
        return List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{resultAliasRow("registry", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", resultFile), resultAliasRow("summary", "resultDashboardTreeCatalogTreeSummaryRegistrySummary:ci", resultFile), resultAliasRow("manifest", "resultDashboardTreeCatalogTreeSummaryRegistryManifest:ci", resultFile), resultAliasRow("envelope", "resultDashboardTreeCatalogTreeSummaryRegistryEnvelope:ci", resultFile), resultAliasRow("snapshot", "resultDashboardTreeCatalogTreeSummaryRegistrySnapshot:ci", resultFile), resultAliasRow("audit", "resultDashboardTreeCatalogTreeSummaryRegistryAudit:ci", resultFile), resultAliasRow("report", "resultDashboardTreeCatalogTreeSummaryRegistryReport:ci", resultFile), resultAliasRow("checklist", "resultDashboardTreeCatalogTreeSummaryRegistryChecklist:ci", resultFile), resultAliasRow("checklistSummary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary:ci", resultFile), resultAliasRow("checklistStatus", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus:ci", resultFile), resultAliasRow("checklistHealth", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth:ci", resultFile), resultAliasRow("checklistBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge:ci", resultFile), resultAliasRow("checklistCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard:ci", resultFile), resultAliasRow("checklistPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel:ci", resultFile), resultAliasRow("checklistWidget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget:ci", resultFile), resultAliasRow("checklistTile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile:ci", resultFile), resultAliasRow("checklistMetric", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric:ci", resultFile), resultAliasRow("checklistTrend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci", resultFile), resultAliasRow("checklistGauge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge:ci", resultFile), resultAliasRow("checklistSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci", resultFile), resultAliasRow("checklistActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar:ci", resultFile), resultAliasRow("checklistDashboard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci", resultFile), resultAliasRow("checklistDashboardManifest", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest:ci", resultFile), resultAliasRow("checklistDashboardSnapshot", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot:ci", resultFile), resultAliasRow("checklistDashboardAudit", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit:ci", resultFile), resultAliasRow("checklistDashboardReport", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport:ci", resultFile), resultAliasRow("checklistDashboardEnvelope", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope:ci", resultFile), resultAliasRow("checklistDashboardPackage", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage:ci", resultFile), resultAliasRow("checklistDashboardBundle", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci", resultFile), resultAliasRow("checklistDashboardDelivery", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci", resultFile), resultAliasRow("checklistDashboardReceipt", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci", resultFile), resultAliasRow("checklistDashboardArchive", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci", resultFile), resultAliasRow("checklistDashboardLedger", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger:ci", resultFile), resultAliasRow("checklistDashboardHistory", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory:ci", resultFile), resultAliasRow("checklistDashboardTimeline", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline:ci", resultFile), resultAliasRow("checklistDashboardTrace", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace:ci", resultFile), resultAliasRow("checklistDashboardTelemetry", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry:ci", resultFile), resultAliasRow("checklistDashboardObservation", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation:ci", resultFile), resultAliasRow("checklistDashboardSignal", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal:ci", resultFile), resultAliasRow("checklistDashboardInsight", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight:ci", resultFile), resultAliasRow("checklistDashboardRecommendation", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation:ci", resultFile), resultAliasRow("checklistDashboardDecision", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision:ci", resultFile), resultAliasRow("checklistDashboardGate", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate:ci", resultFile), resultAliasRow("checklistDashboardGateLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine:ci", resultFile), resultAliasRow("checklistDashboardGateJson", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson:ci", resultFile), resultAliasRow("checklistDashboardGateEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv:ci", resultFile), resultAliasRow("checklistDashboardGateMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateSummary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary:ci", resultFile), resultAliasRow("checklistDashboardGateArtifact", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci", resultFile), resultAliasRow("checklistDashboardGateUploadManifest", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci", resultFile), resultAliasRow("checklistDashboardGateUploadScript", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript:ci", resultFile), resultAliasRow("checklistDashboardGateUploadReadme", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme:ci", resultFile), resultAliasRow("checklistDashboardGateUploadBundle", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci", resultFile), resultAliasRow("checklistDashboardGateUploadIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSummary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsWidget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsTile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsMetric", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsTrend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsGauge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsMenu", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsToolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsCommandPalette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSearchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSearchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsFilterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsViewState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsEmptyState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsLoadingState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsErrorState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsToastState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationCenter", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelMenu", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelViewState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelToastState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadStatus", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus:ci", resultFile), resultAliasRow("checklistDashboardGateUploadStatusLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine:ci", resultFile), resultAliasRow("checklistDetails", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci", resultFile), resultAliasRow("overview", "resultDashboardTreeCatalogTreeSummaryRegistryOverview:ci", resultFile), resultAliasRow("status", "resultDashboardTreeCatalogTreeSummaryRegistryStatus:ci", resultFile), resultAliasRow("health", "resultDashboardTreeCatalogTreeSummaryRegistryHealth:ci", resultFile), resultAliasRow("fields", "resultDashboardTreeCatalogTreeSummaryRegistryFields:ci", resultFile), resultAliasRow("index", "resultDashboardTreeCatalogTreeSummaryRegistryIndex:ci", resultFile), resultAliasRow("sitemap", "resultDashboardTreeCatalogTreeSummaryRegistrySitemap:ci", resultFile), resultAliasRow("breadcrumbs", "resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs:ci", resultFile), resultAliasRow("discovery", "resultDashboardTreeCatalogTreeSummaryRegistryDiscovery:ci", resultFile), resultAliasRow("navigation", "resultDashboardTreeCatalogTreeSummaryRegistryNavigation:ci", resultFile)});
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryCommandSummary(String resultFile) {
        List<String> commands = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummaryRegistryCommandRows(resultFile)) {
            commands.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryCommands", "commandCount", Integer.valueOf(commands.size()), "commands", commands);
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryRegistrySelectorRows(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryCommandRows(resultFile);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistrySelectorSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummaryRegistrySelectorRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistrySelectors", "selectorCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryRegistryAliasRows(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryCommandRows(resultFile);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryAliasSummary(String resultFile) {
        List<String> aliases = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummaryRegistryAliasRows(resultFile)) {
            aliases.add(String.valueOf(row.getOrDefault("name", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryAliases", "aliasCount", Integer.valueOf(aliases.size()), "aliases", aliases);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryDiscovery(String resultFile, Map<String, Object> result) {
        Map<String, Object> discovery = new LinkedHashMap<>();
        discovery.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryDiscovery");
        discovery.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        discovery.put("resultFile", resultFile);
        discovery.put("fields", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryFieldsCi", resultDashboardTreeCatalogTreeSummaryRegistryFieldSummary()));
        discovery.put("commands", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryCommandsCi", resultDashboardTreeCatalogTreeSummaryRegistryCommandSummary(resultFile)));
        discovery.put("selectors", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySelectorsCi", resultDashboardTreeCatalogTreeSummaryRegistrySelectorSummary(resultFile)));
        discovery.put("aliases", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryAliasesCi", resultDashboardTreeCatalogTreeSummaryRegistryAliasSummary(resultFile)));
        return discovery;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryDiscoverySummary(String resultFile) {
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryDiscovery", "fieldCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryRegistryFieldRows().size()), "commandCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryRegistryCommandRows(resultFile).size()), "selectorCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryRegistrySelectorRows(resultFile).size()), "aliasCount", Integer.valueOf(resultDashboardTreeCatalogTreeSummaryRegistryAliasRows(resultFile).size()), "sections", List.of("resultDashboardTreeCatalogTreeSummaryRegistryFields:ci", "resultDashboardTreeCatalogTreeSummaryRegistryCommands:ci", "resultDashboardTreeCatalogTreeSummaryRegistrySelectors:ci", "resultDashboardTreeCatalogTreeSummaryRegistryAliases:ci"));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryNavigation(String resultFile, Map<String, Object> result) {
        Map<String, Object> navigation = new LinkedHashMap<>();
        navigation.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryNavigation");
        navigation.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        navigation.put("resultFile", resultFile);
        navigation.put("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci");
        navigation.put("parent", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryNavigationCi", resultDashboardTreeCatalogTreeSummaryNavigationSummary(resultFile)));
        navigation.put("discovery", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryCi", resultDashboardTreeCatalogTreeSummaryRegistryDiscoverySummary(resultFile)));
        return navigation;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryNavigationSummary(String resultFile) {
        int size;
        Map<String, Object> parent = resultDashboardTreeCatalogTreeSummaryNavigationSummary(resultFile);
        Map<String, Object> discovery = resultDashboardTreeCatalogTreeSummaryRegistryDiscoverySummary(resultFile);
        Object sections = discovery.getOrDefault("sections", List.of());
        if (sections instanceof List) {
            List<?> list = (List) sections;
            size = list.size();
        } else {
            size = 0;
        }
        int discoverySectionCount = size;
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryNavigation", "current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", "parent", "resultDashboardTreeCatalogTreeSummaryNavigation:ci", "parentSectionCount", parent.getOrDefault("sectionCount", 0), "discoverySectionCount", Integer.valueOf(discoverySectionCount), "sections", sections);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs(String resultFile, Map<String, Object> result) {
        Map<String, Object> breadcrumbs = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(resultFile));
        breadcrumbs.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        breadcrumbs.put("resultFile", resultFile);
        breadcrumbs.put("parent", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryBreadcrumbsCi", resultDashboardTreeCatalogTreeSummaryBreadcrumbs(resultFile)));
        return breadcrumbs;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(String resultFile) {
        Map<String, Object> parent = resultDashboardTreeCatalogTreeSummaryBreadcrumbs(resultFile);
        String parentBreadcrumb = String.valueOf(parent.getOrDefault("breadcrumb", "result/dashboard/tree/catalog/treeSummary"));
        int parentDepth = numberAsInt(parent.getOrDefault("depth", 5));
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs", "current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", "parent", "resultDashboardTreeCatalogTreeSummaryBreadcrumbs:ci", "depth", Integer.valueOf(parentDepth + 1), "breadcrumb", parentBreadcrumb + "/registry");
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryRegistrySitemapRows(String resultFile) {
        return List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{resultAliasRow("registry", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", resultFile), resultAliasRow("summary", "resultDashboardTreeCatalogTreeSummaryRegistrySummary:ci", resultFile), resultAliasRow("manifest", "resultDashboardTreeCatalogTreeSummaryRegistryManifest:ci", resultFile), resultAliasRow("envelope", "resultDashboardTreeCatalogTreeSummaryRegistryEnvelope:ci", resultFile), resultAliasRow("snapshot", "resultDashboardTreeCatalogTreeSummaryRegistrySnapshot:ci", resultFile), resultAliasRow("audit", "resultDashboardTreeCatalogTreeSummaryRegistryAudit:ci", resultFile), resultAliasRow("report", "resultDashboardTreeCatalogTreeSummaryRegistryReport:ci", resultFile), resultAliasRow("checklist", "resultDashboardTreeCatalogTreeSummaryRegistryChecklist:ci", resultFile), resultAliasRow("checklistSummary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary:ci", resultFile), resultAliasRow("checklistStatus", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus:ci", resultFile), resultAliasRow("checklistHealth", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth:ci", resultFile), resultAliasRow("checklistBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge:ci", resultFile), resultAliasRow("checklistCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard:ci", resultFile), resultAliasRow("checklistPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel:ci", resultFile), resultAliasRow("checklistWidget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget:ci", resultFile), resultAliasRow("checklistTile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile:ci", resultFile), resultAliasRow("checklistMetric", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric:ci", resultFile), resultAliasRow("checklistTrend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci", resultFile), resultAliasRow("checklistGauge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge:ci", resultFile), resultAliasRow("checklistSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci", resultFile), resultAliasRow("checklistActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar:ci", resultFile), resultAliasRow("checklistDashboard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci", resultFile), resultAliasRow("checklistDashboardManifest", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest:ci", resultFile), resultAliasRow("checklistDashboardSnapshot", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot:ci", resultFile), resultAliasRow("checklistDashboardAudit", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit:ci", resultFile), resultAliasRow("checklistDashboardReport", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport:ci", resultFile), resultAliasRow("checklistDashboardEnvelope", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope:ci", resultFile), resultAliasRow("checklistDashboardPackage", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage:ci", resultFile), resultAliasRow("checklistDashboardBundle", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci", resultFile), resultAliasRow("checklistDashboardDelivery", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci", resultFile), resultAliasRow("checklistDashboardReceipt", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci", resultFile), resultAliasRow("checklistDashboardArchive", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci", resultFile), resultAliasRow("checklistDashboardLedger", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger:ci", resultFile), resultAliasRow("checklistDashboardHistory", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory:ci", resultFile), resultAliasRow("checklistDashboardTimeline", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline:ci", resultFile), resultAliasRow("checklistDashboardTrace", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace:ci", resultFile), resultAliasRow("checklistDashboardTelemetry", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry:ci", resultFile), resultAliasRow("checklistDashboardObservation", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation:ci", resultFile), resultAliasRow("checklistDashboardSignal", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal:ci", resultFile), resultAliasRow("checklistDashboardInsight", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight:ci", resultFile), resultAliasRow("checklistDashboardRecommendation", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation:ci", resultFile), resultAliasRow("checklistDashboardDecision", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision:ci", resultFile), resultAliasRow("checklistDashboardGate", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate:ci", resultFile), resultAliasRow("checklistDashboardGateLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine:ci", resultFile), resultAliasRow("checklistDashboardGateJson", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson:ci", resultFile), resultAliasRow("checklistDashboardGateEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv:ci", resultFile), resultAliasRow("checklistDashboardGateMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateSummary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary:ci", resultFile), resultAliasRow("checklistDashboardGateArtifact", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci", resultFile), resultAliasRow("checklistDashboardGateUploadManifest", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci", resultFile), resultAliasRow("checklistDashboardGateUploadScript", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript:ci", resultFile), resultAliasRow("checklistDashboardGateUploadReadme", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme:ci", resultFile), resultAliasRow("checklistDashboardGateUploadBundle", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci", resultFile), resultAliasRow("checklistDashboardGateUploadIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSummary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsWidget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsTile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsMetric", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsTrend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsGauge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsMenu", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsToolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsCommandPalette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSearchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsSearchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsFilterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsViewState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsEmptyState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsLoadingState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsErrorState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsToastState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationCenter", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelMenu", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsNotificationPanelViewState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci", resultFile), resultAliasRow("checklistDashboardGateUploadFilesChecksumsEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadStatus", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus:ci", resultFile), resultAliasRow("checklistDashboardGateUploadStatusLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv:ci", resultFile), resultAliasRow("checklistDashboardGateUploadJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine:ci", resultFile), resultAliasRow("checklistDashboardGateUploadTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine:ci", resultFile), resultAliasRow("checklistDetails", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci", resultFile), resultAliasRow("overview", "resultDashboardTreeCatalogTreeSummaryRegistryOverview:ci", resultFile), resultAliasRow("status", "resultDashboardTreeCatalogTreeSummaryRegistryStatus:ci", resultFile), resultAliasRow("health", "resultDashboardTreeCatalogTreeSummaryRegistryHealth:ci", resultFile), resultAliasRow("index", "resultDashboardTreeCatalogTreeSummaryRegistryIndex:ci", resultFile), resultAliasRow("fields", "resultDashboardTreeCatalogTreeSummaryRegistryFields:ci", resultFile), resultAliasRow("commands", "resultDashboardTreeCatalogTreeSummaryRegistryCommands:ci", resultFile), resultAliasRow("selectors", "resultDashboardTreeCatalogTreeSummaryRegistrySelectors:ci", resultFile), resultAliasRow("aliases", "resultDashboardTreeCatalogTreeSummaryRegistryAliases:ci", resultFile), resultAliasRow("discovery", "resultDashboardTreeCatalogTreeSummaryRegistryDiscovery:ci", resultFile), resultAliasRow("navigation", "resultDashboardTreeCatalogTreeSummaryRegistryNavigation:ci", resultFile), resultAliasRow("breadcrumbs", "resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs:ci", resultFile), resultAliasRow("sitemap", "resultDashboardTreeCatalogTreeSummaryRegistrySitemap:ci", resultFile)});
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogTreeSummaryRegistrySitemapRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistrySitemap", "endpointCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static List<String> resultDashboardTreeCatalogTreeSummaryRegistrySelfIndex() {
        return List.<String>of((String[]) new String[]{"resultDashboardTreeCatalogTreeSummaryRegistry:ci", "resultDashboardTreeCatalogTreeSummaryRegistrySummary:ci", "resultDashboardTreeCatalogTreeSummaryRegistryManifest:ci", "resultDashboardTreeCatalogTreeSummaryRegistryEnvelope:ci", "resultDashboardTreeCatalogTreeSummaryRegistrySnapshot:ci", "resultDashboardTreeCatalogTreeSummaryRegistryAudit:ci", "resultDashboardTreeCatalogTreeSummaryRegistryReport:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklist:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci", "resultDashboardTreeCatalogTreeSummaryRegistryOverview:ci", "resultDashboardTreeCatalogTreeSummaryRegistryIndex:ci"});
    }

    private static List<String> resultDashboardTreeCatalogTreeSummaryRegistryMetadataIndex() {
        return List.of("resultDashboardTreeCatalogTreeSummaryRegistryFields:ci", "resultDashboardTreeCatalogTreeSummaryRegistryCommands:ci", "resultDashboardTreeCatalogTreeSummaryRegistrySelectors:ci", "resultDashboardTreeCatalogTreeSummaryRegistryAliases:ci");
    }

    private static List<String> resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryIndex() {
        return List.of("resultDashboardTreeCatalogTreeSummaryRegistryDiscovery:ci", "resultDashboardTreeCatalogTreeSummaryRegistryStatus:ci", "resultDashboardTreeCatalogTreeSummaryRegistryHealth:ci", "resultDashboardTreeCatalogTreeSummaryRegistryNavigation:ci", "resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbs:ci", "resultDashboardTreeCatalogTreeSummaryRegistrySitemap:ci");
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryIndex(String resultFile) {
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryIndex", "resultFile", resultFile, "self", resultDashboardTreeCatalogTreeSummaryRegistrySelfIndex(), "metadata", resultDashboardTreeCatalogTreeSummaryRegistryMetadataIndex(), "discovery", resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryIndex());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(String resultFile) {
        List<String> self = resultDashboardTreeCatalogTreeSummaryRegistrySelfIndex();
        List<String> metadata = resultDashboardTreeCatalogTreeSummaryRegistryMetadataIndex();
        List<String> discovery = resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryIndex();
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryIndex", "groupCount", 3, "endpointCount", Integer.valueOf(self.size() + metadata.size() + discovery.size()), "groups", List.of("self", "metadata", "discovery"), "selfCount", Integer.valueOf(self.size()), "metadataCount", Integer.valueOf(metadata.size()), "discoveryCount", Integer.valueOf(discovery.size()));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryOverview(String resultFile, Map<String, Object> result) {
        Map<String, Object> overview = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(resultFile));
        overview.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        overview.put("resultFile", resultFile);
        overview.put("index", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryIndexCi", resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(resultFile)));
        overview.put("sitemap", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySitemapCi", resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(resultFile)));
        overview.put("breadcrumbs", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsCi", resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(resultFile)));
        return overview;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(String resultFile) {
        Map<String, Object> index = resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(resultFile);
        Map<String, Object> sitemap = resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(resultFile);
        Map<String, Object> breadcrumbs = resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(resultFile);
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryOverview", "current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", "groupCount", index.getOrDefault("groupCount", 0), "groupedEndpointCount", index.getOrDefault("endpointCount", 0), "flatEndpointCount", sitemap.getOrDefault("endpointCount", 0), "breadcrumb", breadcrumbs.getOrDefault("breadcrumb", ""));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryStatus(String resultFile, Map<String, Object> result) {
        Map<String, Object> status = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(resultFile));
        status.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        status.put("resultFile", resultFile);
        status.put("overview", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryOverviewCi", resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(resultFile)));
        return status;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(String resultFile) {
        Map<String, Object> overview = resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(resultFile);
        int groupedEndpointCount = numberAsInt(overview.getOrDefault("groupedEndpointCount", 0));
        int flatEndpointCount = numberAsInt(overview.getOrDefault("flatEndpointCount", 0));
        boolean balancedEndpointCounts = groupedEndpointCount == flatEndpointCount;
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryStatus", "status", balancedEndpointCounts ? "OK" : "WARN", "current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", "groupedEndpointCount", Integer.valueOf(groupedEndpointCount), "flatEndpointCount", Integer.valueOf(flatEndpointCount), "balancedEndpointCounts", Boolean.valueOf(balancedEndpointCounts), "breadcrumb", overview.getOrDefault("breadcrumb", ""));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryHealth(String resultFile, Map<String, Object> result) {
        Map<String, Object> health = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryHealthSummary(resultFile));
        health.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        health.put("resultFile", resultFile);
        health.put("status", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryStatusCi", resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(resultFile)));
        return health;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryHealthSummary(String resultFile) {
        Map<String, Object> status = resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(resultFile);
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryHealth", "healthStatus", status.getOrDefault("status", "WARN"), "current", status.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"), "balancedEndpointCounts", status.getOrDefault("balancedEndpointCounts", false), "groupedEndpointCount", status.getOrDefault("groupedEndpointCount", 0), "flatEndpointCount", status.getOrDefault("flatEndpointCount", 0), "breadcrumb", status.getOrDefault("breadcrumb", ""));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistrySummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> summary = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistrySummarySummary(resultFile));
        summary.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        summary.put("resultFile", resultFile);
        summary.put("overview", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryOverviewCi", resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(resultFile)));
        summary.put("status", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryStatusCi", resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(resultFile)));
        summary.put("health", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryHealthCi", resultDashboardTreeCatalogTreeSummaryRegistryHealthSummary(resultFile)));
        return summary;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistrySummarySummary(String resultFile) {
        Map<String, Object> overview = resultDashboardTreeCatalogTreeSummaryRegistryOverviewSummary(resultFile);
        Map<String, Object> status = resultDashboardTreeCatalogTreeSummaryRegistryStatusSummary(resultFile);
        Map<String, Object> health = resultDashboardTreeCatalogTreeSummaryRegistryHealthSummary(resultFile);
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistrySummary", "current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", "groupCount", overview.getOrDefault("groupCount", 0), "groupedEndpointCount", overview.getOrDefault("groupedEndpointCount", 0), "flatEndpointCount", overview.getOrDefault("flatEndpointCount", 0), "status", status.getOrDefault("status", "WARN"), "healthStatus", health.getOrDefault("healthStatus", "WARN"), "breadcrumb", overview.getOrDefault("breadcrumb", ""));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryManifest(String resultFile, Map<String, Object> result) {
        Map<String, Object> manifest = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryManifestSummary(resultFile));
        manifest.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        manifest.put("resultFile", resultFile);
        manifest.put("summary", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySummaryCi", resultDashboardTreeCatalogTreeSummaryRegistrySummarySummary(resultFile)));
        manifest.put("commands", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryCommandsCi", resultDashboardTreeCatalogTreeSummaryRegistryCommandSummary(resultFile)));
        manifest.put("selectors", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySelectorsCi", resultDashboardTreeCatalogTreeSummaryRegistrySelectorSummary(resultFile)));
        return manifest;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryManifestSummary(String resultFile) {
        Map<String, Object> summary = resultDashboardTreeCatalogTreeSummaryRegistrySummarySummary(resultFile);
        Map<String, Object> commands = resultDashboardTreeCatalogTreeSummaryRegistryCommandSummary(resultFile);
        Map<String, Object> selectors = resultDashboardTreeCatalogTreeSummaryRegistrySelectorSummary(resultFile);
        return Map.of("summary", "resultDashboardTreeCatalogTreeSummaryRegistryManifest", "current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci", "status", summary.getOrDefault("status", "WARN"), "healthStatus", summary.getOrDefault("healthStatus", "WARN"), "groupedEndpointCount", summary.getOrDefault("groupedEndpointCount", 0), "flatEndpointCount", summary.getOrDefault("flatEndpointCount", 0), "commandCount", commands.getOrDefault("commandCount", 0), "selectorCount", selectors.getOrDefault("selectorCount", 0));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryEnvelope(String resultFile, Map<String, Object> result) {
        Map<String, Object> envelope = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeSummary(resultFile));
        envelope.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        envelope.put("resultFile", resultFile);
        envelope.put("manifest", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryManifestCi", resultDashboardTreeCatalogTreeSummaryRegistryManifestSummary(resultFile)));
        envelope.put("discovery", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryDiscoveryCi", resultDashboardTreeCatalogTreeSummaryRegistryDiscoverySummary(resultFile)));
        envelope.put("navigation", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryNavigationCi", resultDashboardTreeCatalogTreeSummaryRegistryNavigationSummary(resultFile)));
        envelope.put("breadcrumbs", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsCi", resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(resultFile)));
        return envelope;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeSummary(String resultFile) {
        int size;
        Map<String, Object> manifest = resultDashboardTreeCatalogTreeSummaryRegistryManifestSummary(resultFile);
        Map<String, Object> discovery = resultDashboardTreeCatalogTreeSummaryRegistryDiscoverySummary(resultFile);
        Map<String, Object> breadcrumbs = resultDashboardTreeCatalogTreeSummaryRegistryBreadcrumbsSummary(resultFile);
        Object sections = discovery.getOrDefault("sections", List.of());
        if (sections instanceof List) {
            List<?> list = (List) sections;
            size = list.size();
        } else {
            size = 0;
        }
        int discoverySectionCount = size;
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryEnvelope");
        summary.put("current", manifest.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        summary.put("status", manifest.getOrDefault("status", "WARN"));
        summary.put("healthStatus", manifest.getOrDefault("healthStatus", "WARN"));
        summary.put("groupedEndpointCount", manifest.getOrDefault("groupedEndpointCount", 0));
        summary.put("flatEndpointCount", manifest.getOrDefault("flatEndpointCount", 0));
        summary.put("commandCount", manifest.getOrDefault("commandCount", 0));
        summary.put("selectorCount", manifest.getOrDefault("selectorCount", 0));
        summary.put("discoverySectionCount", Integer.valueOf(discoverySectionCount));
        summary.put("breadcrumb", breadcrumbs.getOrDefault("breadcrumb", ""));
        summary.put("depth", breadcrumbs.getOrDefault("depth", 0));
        return summary;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistrySnapshot(String resultFile, Map<String, Object> result) {
        Map<String, Object> snapshot = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(resultFile));
        snapshot.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        snapshot.put("resultFile", resultFile);
        snapshot.put("envelope", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeCi", resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeSummary(resultFile)));
        snapshot.put("index", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryIndexCi", resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(resultFile)));
        snapshot.put("sitemap", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySitemapCi", resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(resultFile)));
        return snapshot;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(String resultFile) {
        Map<String, Object> envelope = resultDashboardTreeCatalogTreeSummaryRegistryEnvelopeSummary(resultFile);
        Map<String, Object> index = resultDashboardTreeCatalogTreeSummaryRegistryIndexSummary(resultFile);
        Map<String, Object> sitemap = resultDashboardTreeCatalogTreeSummaryRegistrySitemapSummary(resultFile);
        int indexEndpointCount = numberAsInt(index.getOrDefault("endpointCount", 0));
        int sitemapEndpointCount = numberAsInt(sitemap.getOrDefault("endpointCount", 0));
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistrySnapshot");
        summary.put("current", envelope.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        summary.put("status", envelope.getOrDefault("status", "WARN"));
        summary.put("healthStatus", envelope.getOrDefault("healthStatus", "WARN"));
        summary.put("indexEndpointCount", Integer.valueOf(indexEndpointCount));
        summary.put("sitemapEndpointCount", Integer.valueOf(sitemapEndpointCount));
        summary.put("balancedEndpointCounts", Boolean.valueOf(indexEndpointCount == sitemapEndpointCount));
        summary.put("groupCount", index.getOrDefault("groupCount", 0));
        summary.put("commandCount", envelope.getOrDefault("commandCount", 0));
        summary.put("selectorCount", envelope.getOrDefault("selectorCount", 0));
        summary.put("breadcrumb", envelope.getOrDefault("breadcrumb", ""));
        return summary;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryAudit(String resultFile, Map<String, Object> result) {
        Map<String, Object> audit = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryAuditSummary(resultFile));
        audit.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        audit.put("resultFile", resultFile);
        audit.put("snapshot", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySnapshotCi", resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(resultFile)));
        return audit;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryAuditSummary(String resultFile) {
        Map<String, Object> snapshot = resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(resultFile);
        boolean statusOk = "OK".equals(String.valueOf(snapshot.getOrDefault("status", "WARN")));
        boolean healthOk = "OK".equals(String.valueOf(snapshot.getOrDefault("healthStatus", "WARN")));
        boolean balancedEndpointCounts = Boolean.TRUE.equals(snapshot.getOrDefault("balancedEndpointCounts", false));
        boolean auditPassed = statusOk && healthOk && balancedEndpointCounts;
        Map<String, Object> audit = new LinkedHashMap<>();
        audit.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryAudit");
        audit.put("current", snapshot.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        audit.put("auditPassed", Boolean.valueOf(auditPassed));
        audit.put("auditStatus", auditPassed ? "PASS" : "FAIL");
        audit.put("reason", auditPassed ? "registry snapshot is healthy and endpoint counts are balanced" : "registry snapshot needs attention");
        audit.put("status", snapshot.getOrDefault("status", "WARN"));
        audit.put("healthStatus", snapshot.getOrDefault("healthStatus", "WARN"));
        audit.put("balancedEndpointCounts", Boolean.valueOf(balancedEndpointCounts));
        audit.put("indexEndpointCount", snapshot.getOrDefault("indexEndpointCount", 0));
        audit.put("sitemapEndpointCount", snapshot.getOrDefault("sitemapEndpointCount", 0));
        audit.put("commandCount", snapshot.getOrDefault("commandCount", 0));
        audit.put("selectorCount", snapshot.getOrDefault("selectorCount", 0));
        audit.put("breadcrumb", snapshot.getOrDefault("breadcrumb", ""));
        return audit;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryReport(String resultFile, Map<String, Object> result) {
        Map<String, Object> report = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(resultFile));
        report.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        report.put("resultFile", resultFile);
        report.put("audit", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryAuditCi", resultDashboardTreeCatalogTreeSummaryRegistryAuditSummary(resultFile)));
        report.put("snapshot", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistrySnapshotCi", resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(resultFile)));
        return report;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(String resultFile) {
        Map<String, Object> audit = resultDashboardTreeCatalogTreeSummaryRegistryAuditSummary(resultFile);
        Map<String, Object> snapshot = resultDashboardTreeCatalogTreeSummaryRegistrySnapshotSummary(resultFile);
        boolean auditPassed = Boolean.TRUE.equals(audit.getOrDefault("auditPassed", false));
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryReport");
        report.put("current", audit.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        report.put("reportStatus", auditPassed ? "READY" : "NEEDS_ATTENTION");
        report.put("auditStatus", audit.getOrDefault("auditStatus", "FAIL"));
        report.put("auditPassed", Boolean.valueOf(auditPassed));
        report.put("status", audit.getOrDefault("status", "WARN"));
        report.put("healthStatus", audit.getOrDefault("healthStatus", "WARN"));
        report.put("balancedEndpointCounts", audit.getOrDefault("balancedEndpointCounts", false));
        report.put("indexEndpointCount", audit.getOrDefault("indexEndpointCount", 0));
        report.put("sitemapEndpointCount", audit.getOrDefault("sitemapEndpointCount", 0));
        report.put("groupCount", snapshot.getOrDefault("groupCount", 0));
        report.put("commandCount", audit.getOrDefault("commandCount", 0));
        report.put("selectorCount", audit.getOrDefault("selectorCount", 0));
        report.put("breadcrumb", audit.getOrDefault("breadcrumb", ""));
        return report;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklist(String resultFile, Map<String, Object> result) {
        Map<String, Object> checklist = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile));
        checklist.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        checklist.put("resultFile", resultFile);
        checklist.put("report", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryReportCi", resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(resultFile)));
        checklist.put("checks", resultDashboardTreeCatalogTreeSummaryRegistryChecklistRows(resultFile));
        return checklist;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> summary = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile));
        summary.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        summary.put("resultFile", resultFile);
        summary.put("report", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryReportCi", resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(resultFile)));
        return summary;
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogTreeSummaryRegistryChecklistRows(String resultFile) {
        Map<String, Object> report = resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(resultFile);
        int commandCount = numberAsInt(report.getOrDefault("commandCount", 0));
        int selectorCount = numberAsInt(report.getOrDefault("selectorCount", 0));
        return List.of(resultCheckRow("auditPassed", Boolean.TRUE.equals(report.getOrDefault("auditPassed", false)), "Registry audit passes."), resultCheckRow("statusOk", "OK".equals(String.valueOf(report.getOrDefault("status", "WARN"))), "Registry status is OK."), resultCheckRow("healthOk", "OK".equals(String.valueOf(report.getOrDefault("healthStatus", "WARN"))), "Registry health is OK."), resultCheckRow("balancedEndpointCounts", Boolean.TRUE.equals(report.getOrDefault("balancedEndpointCounts", false)), "Index and sitemap endpoint counts match."), resultCheckRow("commandSelectorParity", commandCount == selectorCount, "Command and selector counts match."));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(String resultFile) {
        List<Map<String, Object>> checks = resultDashboardTreeCatalogTreeSummaryRegistryChecklistRows(resultFile);
        long passedCount = checks.stream().filter(row -> {
            return Boolean.TRUE.equals(row.get("passed"));
        }).count();
        int failedCount = checks.size() - ((int) passedCount);
        Map<String, Object> report = resultDashboardTreeCatalogTreeSummaryRegistryReportSummary(resultFile);
        Map<String, Object> checklist = new LinkedHashMap<>();
        checklist.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklist");
        checklist.put("current", report.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        checklist.put("reportStatus", report.getOrDefault("reportStatus", "NEEDS_ATTENTION"));
        checklist.put("checkCount", Integer.valueOf(checks.size()));
        checklist.put("passedCount", Long.valueOf(passedCount));
        checklist.put("failedCount", Integer.valueOf(failedCount));
        checklist.put("allPassed", Boolean.valueOf(failedCount == 0));
        checklist.put("breadcrumb", report.getOrDefault("breadcrumb", ""));
        return checklist;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus(String resultFile, Map<String, Object> result) {
        Map<String, Object> status = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusSummary(resultFile));
        status.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        status.put("resultFile", resultFile);
        status.put("checklist", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile)));
        return status;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusSummary(String resultFile) {
        Map<String, Object> checklist = resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile);
        boolean allPassed = Boolean.TRUE.equals(checklist.getOrDefault("allPassed", false));
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatus");
        status.put("current", checklist.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        status.put("status", allPassed ? "OK" : "WARN");
        status.put("reportStatus", checklist.getOrDefault("reportStatus", "NEEDS_ATTENTION"));
        status.put("allPassed", Boolean.valueOf(allPassed));
        status.put("checkCount", checklist.getOrDefault("checkCount", 0));
        status.put("passedCount", checklist.getOrDefault("passedCount", 0));
        status.put("failedCount", checklist.getOrDefault("failedCount", 0));
        status.put("breadcrumb", checklist.getOrDefault("breadcrumb", ""));
        return status;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth(String resultFile, Map<String, Object> result) {
        Map<String, Object> health = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthSummary(resultFile));
        health.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        health.put("resultFile", resultFile);
        health.put("status", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusSummary(resultFile)));
        return health;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthSummary(String resultFile) {
        Map<String, Object> status = resultDashboardTreeCatalogTreeSummaryRegistryChecklistStatusSummary(resultFile);
        boolean allPassed = Boolean.TRUE.equals(status.getOrDefault("allPassed", false));
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealth");
        health.put("current", status.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        health.put("health", allPassed ? "healthy" : "degraded");
        health.put("status", status.getOrDefault("status", allPassed ? "OK" : "WARN"));
        health.put("reportStatus", status.getOrDefault("reportStatus", "NEEDS_ATTENTION"));
        health.put("allPassed", Boolean.valueOf(allPassed));
        health.put("failedCount", status.getOrDefault("failedCount", 0));
        health.put("breadcrumb", status.getOrDefault("breadcrumb", ""));
        return health;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge(String resultFile, Map<String, Object> result) {
        Map<String, Object> badge = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeSummary(resultFile));
        badge.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        badge.put("resultFile", resultFile);
        badge.put("health", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthSummary(resultFile)));
        return badge;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeSummary(String resultFile) {
        Map<String, Object> health = resultDashboardTreeCatalogTreeSummaryRegistryChecklistHealthSummary(resultFile);
        boolean allPassed = Boolean.TRUE.equals(health.getOrDefault("allPassed", false));
        Map<String, Object> badge = new LinkedHashMap<>();
        badge.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadge");
        badge.put("current", health.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        badge.put("label", allPassed ? "Checklist Ready" : "Checklist Attention");
        badge.put("severity", allPassed ? "success" : "warning");
        badge.put("color", allPassed ? "green" : "amber");
        badge.put("health", health.getOrDefault("health", allPassed ? "healthy" : "degraded"));
        badge.put("status", health.getOrDefault("status", allPassed ? "OK" : "WARN"));
        badge.put("reportStatus", health.getOrDefault("reportStatus", "NEEDS_ATTENTION"));
        badge.put("failedCount", health.getOrDefault("failedCount", 0));
        badge.put("breadcrumb", health.getOrDefault("breadcrumb", ""));
        return badge;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard(String resultFile, Map<String, Object> result) {
        Map<String, Object> card = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardSummary(resultFile));
        card.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        card.put("resultFile", resultFile);
        card.put("badge", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeSummary(resultFile)));
        return card;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardSummary(String resultFile) {
        Map<String, Object> badge = resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeSummary(resultFile);
        boolean ready = "success".equals(String.valueOf(badge.getOrDefault("severity", "")));
        Map<String, Object> card = new LinkedHashMap<>();
        card.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard");
        card.put("current", badge.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        card.put("title", "Registry Checklist");
        card.put("subtitle", ready ? "All registry readiness checks are passing." : "Registry checklist needs attention.");
        card.put("primaryAction", "Open checklist details");
        card.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci");
        card.put("badgeLabel", badge.getOrDefault("label", ready ? "Checklist Ready" : "Checklist Attention"));
        card.put("severity", badge.getOrDefault("severity", ready ? "success" : "warning"));
        card.put("health", badge.getOrDefault("health", ready ? "healthy" : "degraded"));
        card.put("status", badge.getOrDefault("status", ready ? "OK" : "WARN"));
        card.put("failedCount", badge.getOrDefault("failedCount", 0));
        card.put("breadcrumb", badge.getOrDefault("breadcrumb", ""));
        return card;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel(String resultFile, Map<String, Object> result) {
        Map<String, Object> panel = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelSummary(resultFile));
        panel.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        panel.put("resultFile", resultFile);
        panel.put("card", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardSummary(resultFile)));
        panel.put("badge", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistBadgeSummary(resultFile)));
        return panel;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelSummary(String resultFile) {
        Map<String, Object> card = resultDashboardTreeCatalogTreeSummaryRegistryChecklistCardSummary(resultFile);
        Map<String, Object> panel = new LinkedHashMap<>();
        panel.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel");
        panel.put("current", card.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        panel.put("panel", "registryChecklist");
        panel.put("title", card.getOrDefault("title", "Registry Checklist"));
        panel.put("subtitle", card.getOrDefault("subtitle", ""));
        panel.put("badgeLabel", card.getOrDefault("badgeLabel", ""));
        panel.put("severity", card.getOrDefault("severity", "warning"));
        panel.put("primarySelector", card.getOrDefault("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci"));
        panel.put("secondarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistCard:ci");
        panel.put("detailsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci");
        panel.put("status", card.getOrDefault("status", "WARN"));
        panel.put("failedCount", card.getOrDefault("failedCount", 0));
        panel.put("breadcrumb", card.getOrDefault("breadcrumb", ""));
        return panel;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget(String resultFile, Map<String, Object> result) {
        Map<String, Object> widget = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetSummary(resultFile));
        widget.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        widget.put("resultFile", resultFile);
        widget.put("panel", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelSummary(resultFile)));
        return widget;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetSummary(String resultFile) {
        Map<String, Object> panel = resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanelSummary(resultFile);
        Map<String, Object> widget = new LinkedHashMap<>();
        widget.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget");
        widget.put("current", panel.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        widget.put("widget", "registryChecklist");
        widget.put("label", panel.getOrDefault("badgeLabel", "Checklist Ready"));
        widget.put("state", panel.getOrDefault("status", "WARN"));
        widget.put("severity", panel.getOrDefault("severity", "warning"));
        widget.put("targetSelector", panel.getOrDefault("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci"));
        widget.put("panelSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistPanel:ci");
        widget.put("failedCount", panel.getOrDefault("failedCount", 0));
        widget.put("breadcrumb", panel.getOrDefault("breadcrumb", ""));
        return widget;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile(String resultFile, Map<String, Object> result) {
        Map<String, Object> tile = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileSummary(resultFile));
        tile.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        tile.put("resultFile", resultFile);
        tile.put("widget", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetSummary(resultFile)));
        return tile;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileSummary(String resultFile) {
        Map<String, Object> widget = resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidgetSummary(resultFile);
        String severity = String.valueOf(widget.getOrDefault("severity", "warning"));
        Map<String, Object> tile = new LinkedHashMap<>();
        tile.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTile");
        tile.put("current", widget.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        tile.put("tile", "registryChecklist");
        tile.put("title", "Registry Checklist");
        tile.put("value", widget.getOrDefault("label", "Checklist Ready"));
        tile.put("tone", "success".equals(severity) ? "positive" : "attention");
        tile.put("state", widget.getOrDefault("state", "WARN"));
        tile.put("severity", severity);
        tile.put("actionSelector", widget.getOrDefault("targetSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci"));
        tile.put("widgetSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistWidget:ci");
        tile.put("failedCount", widget.getOrDefault("failedCount", 0));
        tile.put("breadcrumb", widget.getOrDefault("breadcrumb", ""));
        return tile;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric(String resultFile, Map<String, Object> result) {
        Map<String, Object> metric = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricSummary(resultFile));
        metric.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        metric.put("resultFile", resultFile);
        metric.put("tile", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistTileSummary(resultFile)));
        return metric;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricSummary(String resultFile) {
        Map<String, Object> checklist = resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile);
        boolean allPassed = Boolean.TRUE.equals(checklist.getOrDefault("allPassed", false));
        int checkCount = numberAsInt(checklist.getOrDefault("checkCount", 0));
        int passedCount = numberAsInt(checklist.getOrDefault("passedCount", 0));
        int failedCount = numberAsInt(checklist.getOrDefault("failedCount", 0));
        Map<String, Object> metric = new LinkedHashMap<>();
        metric.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric");
        metric.put("current", checklist.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        metric.put("metric", "registryChecklistReadiness");
        metric.put("value", passedCount);
        metric.put("max", checkCount);
        metric.put("unit", "checks");
        metric.put("ratio", checkCount == 0 ? 0.0 : ((double) passedCount) / checkCount);
        metric.put("status", allPassed ? "OK" : "WARN");
        metric.put("failedCount", failedCount);
        metric.put("targetSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci");
        metric.put("breadcrumb", checklist.getOrDefault("breadcrumb", ""));
        return metric;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend(String resultFile, Map<String, Object> result) {
        Map<String, Object> trend = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendSummary(resultFile));
        trend.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        trend.put("resultFile", resultFile);
        trend.put("metric", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricSummary(resultFile)));
        return trend;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendSummary(String resultFile) {
        Map<String, Object> metric = resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricSummary(resultFile);
        String pointId = "registryChecklistReadiness@" + resultFile;
        Map<String, Object> point = new LinkedHashMap<>();
        point.put("id", pointId);
        point.put("label", "registryChecklistReadiness");
        point.put("value", metric.getOrDefault("value", 0));
        point.put("max", metric.getOrDefault("max", 0));
        point.put("ratio", metric.getOrDefault("ratio", 0.0));
        point.put("status", metric.getOrDefault("status", "WARN"));
        point.put("sourceSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric:ci");
        Map<String, Object> trend = new LinkedHashMap<>();
        trend.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend");
        trend.put("current", metric.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        trend.put("series", "registryChecklistReadiness");
        trend.put("pointCount", 1);
        trend.put("points", List.<Map<String, Object>>of((Map<String, Object>) point));
        trend.put("latestValue", metric.getOrDefault("value", 0));
        trend.put("latestRatio", metric.getOrDefault("ratio", 0.0));
        trend.put("status", metric.getOrDefault("status", "WARN"));
        trend.put("targetSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric:ci");
        trend.put("breadcrumb", metric.getOrDefault("breadcrumb", ""));
        return trend;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge(String resultFile, Map<String, Object> result) {
        Map<String, Object> gauge = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeSummary(resultFile));
        gauge.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        gauge.put("resultFile", resultFile);
        gauge.put("trend", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendSummary(resultFile)));
        return gauge;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeSummary(String resultFile) {
        Map<String, Object> metric = resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetricSummary(resultFile);
        int value = numberAsInt(metric.getOrDefault("value", 0));
        int max = numberAsInt(metric.getOrDefault("max", 0));
        double ratio = asDouble(metric.getOrDefault("ratio", 0.0));
        String status = String.valueOf(metric.getOrDefault("status", "WARN"));
        Map<String, Object> gauge = new LinkedHashMap<>();
        gauge.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge");
        gauge.put("current", metric.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        gauge.put("label", "Registry checklist readiness");
        gauge.put("value", value);
        gauge.put("min", 0);
        gauge.put("max", max);
        gauge.put("unit", metric.getOrDefault("unit", "checks"));
        gauge.put("ratio", ratio);
        gauge.put("percent", ratio * 100.0);
        gauge.put("status", status);
        gauge.put("tone", "OK".equals(status) ? "positive" : "attention");
        gauge.put("failedCount", metric.getOrDefault("failedCount", 0));
        gauge.put("metricSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistMetric:ci");
        gauge.put("trendSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci");
        gauge.put("targetSelector", metric.getOrDefault("targetSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci"));
        gauge.put("breadcrumb", metric.getOrDefault("breadcrumb", ""));
        return gauge;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard(String resultFile, Map<String, Object> result) {
        Map<String, Object> card = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(resultFile));
        card.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        card.put("resultFile", resultFile);
        card.put("gauge", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeSummary(resultFile)));
        card.put("trend", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrendSummary(resultFile)));
        return card;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(String resultFile) {
        Map<String, Object> gauge = resultDashboardTreeCatalogTreeSummaryRegistryChecklistGaugeSummary(resultFile);
        Map<String, Object> details = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsSummary(resultFile);
        Map<String, Object> card = new LinkedHashMap<>();
        card.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard");
        card.put("current", gauge.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        card.put("title", "Registry checklist readiness");
        card.put("value", gauge.getOrDefault("value", 0));
        card.put("max", gauge.getOrDefault("max", 0));
        card.put("percent", gauge.getOrDefault("percent", 0.0));
        card.put("status", gauge.getOrDefault("status", "WARN"));
        card.put("tone", gauge.getOrDefault("tone", "attention"));
        card.put("failedCount", gauge.getOrDefault("failedCount", 0));
        card.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge:ci");
        card.put("trendSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci");
        card.put("detailsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci");
        card.put("checkCount", details.getOrDefault("checkCount", 0));
        card.put("action", "Open checklist details");
        card.put("breadcrumb", gauge.getOrDefault("breadcrumb", ""));
        return card;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar(String resultFile, Map<String, Object> result) {
        Map<String, Object> actionBar = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarSummary(resultFile));
        actionBar.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        actionBar.put("resultFile", resultFile);
        actionBar.put("summaryCard", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(resultFile)));
        return actionBar;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarSummary(String resultFile) {
        Map<String, Object> card = resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(resultFile);
        String tone = String.valueOf(card.getOrDefault("tone", "attention"));
        Map<String, Object> actionBar = new LinkedHashMap<>();
        actionBar.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar");
        actionBar.put("current", card.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        actionBar.put("label", "Registry checklist actions");
        actionBar.put("actionCount", 4);
        actionBar.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci");
        actionBar.put("actions", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{checklistAction("summaryCard", "Summary card", "primary", tone, "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci"), checklistAction("gauge", "Open gauge", "secondary", tone, "resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge:ci"), checklistAction("trend", "Open trend", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci"), checklistAction("details", "Open details", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci")}));
        actionBar.put("status", card.getOrDefault("status", "WARN"));
        actionBar.put("tone", tone);
        actionBar.put("breadcrumb", card.getOrDefault("breadcrumb", ""));
        return actionBar;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard(String resultFile, Map<String, Object> result) {
        Map<String, Object> dashboard = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(resultFile));
        dashboard.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        dashboard.put("resultFile", resultFile);
        dashboard.put("summaryCard", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(resultFile)));
        dashboard.put("actionBar", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarSummary(resultFile)));
        return dashboard;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(String resultFile) {
        Map<String, Object> card = resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCardSummary(resultFile);
        Map<String, Object> actionBar = resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBarSummary(resultFile);
        Map<String, Object> dashboard = new LinkedHashMap<>();
        dashboard.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard");
        dashboard.put("current", card.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        dashboard.put("title", "Registry checklist dashboard");
        dashboard.put("section", "registryChecklist");
        dashboard.put("status", card.getOrDefault("status", "WARN"));
        dashboard.put("tone", card.getOrDefault("tone", "attention"));
        dashboard.put("summaryCardSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci");
        dashboard.put("actionBarSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar:ci");
        dashboard.put("detailsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci");
        dashboard.put("actionCount", actionBar.getOrDefault("actionCount", 0));
        dashboard.put("value", card.getOrDefault("value", 0));
        dashboard.put("max", card.getOrDefault("max", 0));
        dashboard.put("percent", card.getOrDefault("percent", 0.0));
        dashboard.put("breadcrumb", card.getOrDefault("breadcrumb", ""));
        return dashboard;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest(String resultFile, Map<String, Object> result) {
        Map<String, Object> manifest = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestSummary(resultFile));
        manifest.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        manifest.put("resultFile", resultFile);
        manifest.put("dashboard", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(resultFile)));
        return manifest;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestSummary(String resultFile) {
        Map<String, Object> dashboard = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(resultFile);
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest");
        manifest.put("current", dashboard.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        manifest.put("section", dashboard.getOrDefault("section", "registryChecklist"));
        manifest.put("schemaVersion", "checklist-dashboard/v1");
        manifest.put("status", dashboard.getOrDefault("status", "WARN"));
        manifest.put("fields", List.<String>of((String[]) new String[]{"title", "section", "status", "tone", "summaryCardSelector", "actionBarSelector", "detailsSelector", "actionCount", "value", "max", "percent", "breadcrumb"}));
        manifest.put("fieldCount", 12);
        manifest.put("selectors", List.<String>of((String[]) new String[]{"resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummaryCard:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistActionBar:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistGauge:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistTrend:ci", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci"}));
        manifest.put("selectorCount", 6);
        manifest.put("targetSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci");
        manifest.put("breadcrumb", dashboard.getOrDefault("breadcrumb", ""));
        return manifest;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot(String resultFile, Map<String, Object> result) {
        Map<String, Object> snapshot = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(resultFile));
        snapshot.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        snapshot.put("resultFile", resultFile);
        snapshot.put("dashboard", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(resultFile)));
        snapshot.put("manifest", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestSummary(resultFile)));
        return snapshot;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(String resultFile) {
        Map<String, Object> dashboard = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(resultFile);
        Map<String, Object> manifest = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestSummary(resultFile);
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot");
        snapshot.put("current", dashboard.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        snapshot.put("snapshotVersion", "checklist-dashboard-snapshot/v1");
        snapshot.put("schemaVersion", manifest.getOrDefault("schemaVersion", "checklist-dashboard/v1"));
        snapshot.put("section", dashboard.getOrDefault("section", "registryChecklist"));
        snapshot.put("status", dashboard.getOrDefault("status", "WARN"));
        snapshot.put("tone", dashboard.getOrDefault("tone", "attention"));
        snapshot.put("value", dashboard.getOrDefault("value", 0));
        snapshot.put("max", dashboard.getOrDefault("max", 0));
        snapshot.put("percent", dashboard.getOrDefault("percent", 0.0));
        snapshot.put("fieldCount", manifest.getOrDefault("fieldCount", 0));
        snapshot.put("selectorCount", manifest.getOrDefault("selectorCount", 0));
        snapshot.put("dashboardSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci");
        snapshot.put("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest:ci");
        snapshot.put("breadcrumb", dashboard.getOrDefault("breadcrumb", ""));
        return snapshot;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit(String resultFile, Map<String, Object> result) {
        Map<String, Object> audit = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditSummary(resultFile));
        audit.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        audit.put("resultFile", resultFile);
        audit.put("snapshot", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(resultFile)));
        return audit;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditSummary(String resultFile) {
        Map<String, Object> snapshot = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(resultFile);
        boolean ok = "OK".equals(String.valueOf(snapshot.getOrDefault("status", "WARN")));
        Map<String, Object> audit = new LinkedHashMap<>();
        audit.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit");
        audit.put("current", snapshot.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        audit.put("auditVersion", "checklist-dashboard-audit/v1");
        audit.put("snapshotVersion", snapshot.getOrDefault("snapshotVersion", "checklist-dashboard-snapshot/v1"));
        audit.put("schemaVersion", snapshot.getOrDefault("schemaVersion", "checklist-dashboard/v1"));
        audit.put("section", snapshot.getOrDefault("section", "registryChecklist"));
        audit.put("status", ok ? "OK" : "WARN");
        audit.put("verdict", ok ? "pass" : "review");
        audit.put("fieldCount", snapshot.getOrDefault("fieldCount", 0));
        audit.put("selectorCount", snapshot.getOrDefault("selectorCount", 0));
        audit.put("evidenceSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot:ci");
        audit.put("targetSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci");
        audit.put("breadcrumb", snapshot.getOrDefault("breadcrumb", ""));
        return audit;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport(String resultFile, Map<String, Object> result) {
        Map<String, Object> report = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportSummary(resultFile));
        report.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        report.put("resultFile", resultFile);
        report.put("dashboard", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSummary(resultFile)));
        report.put("manifest", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifestSummary(resultFile)));
        report.put("snapshot", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(resultFile)));
        report.put("audit", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditSummary(resultFile)));
        return report;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportSummary(String resultFile) {
        Map<String, Object> audit = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAuditSummary(resultFile);
        Map<String, Object> snapshot = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshotSummary(resultFile);
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport");
        report.put("current", audit.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        report.put("report", "checklistDashboard");
        report.put("status", audit.getOrDefault("status", "WARN"));
        report.put("verdict", audit.getOrDefault("verdict", "review"));
        report.put("section", audit.getOrDefault("section", "registryChecklist"));
        report.put("schemaVersion", audit.getOrDefault("schemaVersion", "checklist-dashboard/v1"));
        report.put("snapshotVersion", audit.getOrDefault("snapshotVersion", "checklist-dashboard-snapshot/v1"));
        report.put("value", snapshot.getOrDefault("value", 0));
        report.put("max", snapshot.getOrDefault("max", 0));
        report.put("percent", snapshot.getOrDefault("percent", 0.0));
        report.put("fieldCount", audit.getOrDefault("fieldCount", 0));
        report.put("selectorCount", audit.getOrDefault("selectorCount", 0));
        report.put("dashboardSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci");
        report.put("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardManifest:ci");
        report.put("snapshotSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSnapshot:ci");
        report.put("auditSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit:ci");
        report.put("breadcrumb", audit.getOrDefault("breadcrumb", ""));
        return report;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope(String resultFile, Map<String, Object> result) {
        Map<String, Object> envelope = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeSummary(resultFile));
        envelope.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        envelope.put("resultFile", resultFile);
        envelope.put("reportPayload", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportSummary(resultFile)));
        return envelope;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeSummary(String resultFile) {
        Map<String, Object> report = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReportSummary(resultFile);
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope");
        envelope.put("current", report.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        envelope.put("envelopeVersion", "checklist-dashboard-envelope/v1");
        envelope.put("payload", "checklistDashboardReport");
        envelope.put("status", report.getOrDefault("status", "WARN"));
        envelope.put("verdict", report.getOrDefault("verdict", "review"));
        envelope.put("section", report.getOrDefault("section", "registryChecklist"));
        envelope.put("reportSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport:ci");
        envelope.put("dashboardSelector", report.getOrDefault("dashboardSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci"));
        envelope.put("auditSelector", report.getOrDefault("auditSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardAudit:ci"));
        envelope.put("breadcrumb", report.getOrDefault("breadcrumb", ""));
        return envelope;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage(String resultFile, Map<String, Object> result) {
        Map<String, Object> pkg = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageSummary(resultFile));
        pkg.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        pkg.put("resultFile", resultFile);
        pkg.put("envelope", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeSummary(resultFile)));
        return pkg;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageSummary(String resultFile) {
        Map<String, Object> envelope = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelopeSummary(resultFile);
        Map<String, Object> pkg = new LinkedHashMap<>();
        pkg.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage");
        pkg.put("current", envelope.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        pkg.put("packageVersion", "checklist-dashboard-package/v1");
        pkg.put("payload", envelope.getOrDefault("payload", "checklistDashboardReport"));
        pkg.put("status", envelope.getOrDefault("status", "WARN"));
        pkg.put("verdict", envelope.getOrDefault("verdict", "review"));
        pkg.put("section", envelope.getOrDefault("section", "registryChecklist"));
        pkg.put("envelopeSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope:ci");
        pkg.put("reportSelector", envelope.getOrDefault("reportSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport:ci"));
        pkg.put("dashboardSelector", envelope.getOrDefault("dashboardSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci"));
        pkg.put("breadcrumb", envelope.getOrDefault("breadcrumb", ""));
        return pkg;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle(String resultFile, Map<String, Object> result) {
        Map<String, Object> bundle = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleSummary(resultFile));
        bundle.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        bundle.put("resultFile", resultFile);
        bundle.put("package", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageSummary(resultFile)));
        bundle.put("details", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsSummary(resultFile)));
        return bundle;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleSummary(String resultFile) {
        Map<String, Object> pkg = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackageSummary(resultFile);
        Map<String, Object> bundle = new LinkedHashMap<>();
        bundle.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle");
        bundle.put("current", pkg.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        bundle.put("bundleVersion", "checklist-dashboard-bundle/v1");
        bundle.put("packageVersion", pkg.getOrDefault("packageVersion", "checklist-dashboard-package/v1"));
        bundle.put("payload", pkg.getOrDefault("payload", "checklistDashboardReport"));
        bundle.put("status", pkg.getOrDefault("status", "WARN"));
        bundle.put("verdict", pkg.getOrDefault("verdict", "review"));
        bundle.put("section", pkg.getOrDefault("section", "registryChecklist"));
        bundle.put("packageSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage:ci");
        bundle.put("detailsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci");
        bundle.put("envelopeSelector", pkg.getOrDefault("envelopeSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardEnvelope:ci"));
        bundle.put("reportSelector", pkg.getOrDefault("reportSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReport:ci"));
        bundle.put("dashboardSelector", pkg.getOrDefault("dashboardSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboard:ci"));
        bundle.put("breadcrumb", pkg.getOrDefault("breadcrumb", ""));
        return bundle;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery(String resultFile, Map<String, Object> result) {
        Map<String, Object> delivery = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliverySummary(resultFile));
        delivery.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        delivery.put("resultFile", resultFile);
        delivery.put("bundle", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleSummary(resultFile)));
        return delivery;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliverySummary(String resultFile) {
        Map<String, Object> bundle = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundleSummary(resultFile);
        Map<String, Object> delivery = new LinkedHashMap<>();
        delivery.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery");
        delivery.put("current", bundle.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        delivery.put("deliveryVersion", "checklist-dashboard-delivery/v1");
        delivery.put("bundleVersion", bundle.getOrDefault("bundleVersion", "checklist-dashboard-bundle/v1"));
        delivery.put("transport", "selector");
        delivery.put("route", "result/dashboard/tree/catalog/treeSummary/registry/checklist/dashboard");
        delivery.put("payload", bundle.getOrDefault("payload", "checklistDashboardReport"));
        delivery.put("status", bundle.getOrDefault("status", "WARN"));
        delivery.put("verdict", bundle.getOrDefault("verdict", "review"));
        delivery.put("section", bundle.getOrDefault("section", "registryChecklist"));
        delivery.put("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci");
        delivery.put("packageSelector", bundle.getOrDefault("packageSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardPackage:ci"));
        delivery.put("detailsSelector", bundle.getOrDefault("detailsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails:ci"));
        delivery.put("breadcrumb", bundle.getOrDefault("breadcrumb", ""));
        return delivery;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt(String resultFile, Map<String, Object> result) {
        Map<String, Object> receipt = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptSummary(resultFile));
        receipt.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        receipt.put("resultFile", resultFile);
        receipt.put("delivery", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliveryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliverySummary(resultFile)));
        return receipt;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptSummary(String resultFile) {
        Map<String, Object> delivery = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDeliverySummary(resultFile);
        Map<String, Object> receipt = new LinkedHashMap<>();
        receipt.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt");
        receipt.put("current", delivery.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        receipt.put("receiptVersion", "checklist-dashboard-receipt/v1");
        receipt.put("deliveryVersion", delivery.getOrDefault("deliveryVersion", "checklist-dashboard-delivery/v1"));
        receipt.put("acknowledgement", "accepted");
        receipt.put("transport", delivery.getOrDefault("transport", "selector"));
        receipt.put("route", delivery.getOrDefault("route", "result/dashboard/tree/catalog/treeSummary/registry/checklist/dashboard"));
        receipt.put("payload", delivery.getOrDefault("payload", "checklistDashboardReport"));
        receipt.put("status", delivery.getOrDefault("status", "WARN"));
        receipt.put("verdict", delivery.getOrDefault("verdict", "review"));
        receipt.put("section", delivery.getOrDefault("section", "registryChecklist"));
        receipt.put("deliverySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci");
        receipt.put("bundleSelector", delivery.getOrDefault("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci"));
        receipt.put("breadcrumb", delivery.getOrDefault("breadcrumb", ""));
        return receipt;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive(String resultFile, Map<String, Object> result) {
        Map<String, Object> archive = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveSummary(resultFile));
        archive.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        archive.put("resultFile", resultFile);
        archive.put("receipt", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptSummary(resultFile)));
        return archive;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveSummary(String resultFile) {
        Map<String, Object> receipt = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceiptSummary(resultFile);
        Map<String, Object> archive = new LinkedHashMap<>();
        archive.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive");
        archive.put("current", receipt.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        archive.put("archiveVersion", "checklist-dashboard-archive/v1");
        archive.put("receiptVersion", receipt.getOrDefault("receiptVersion", "checklist-dashboard-receipt/v1"));
        archive.put("recordState", "archived");
        archive.put("acknowledgement", receipt.getOrDefault("acknowledgement", "accepted"));
        archive.put("transport", receipt.getOrDefault("transport", "selector"));
        archive.put("route", receipt.getOrDefault("route", "result/dashboard/tree/catalog/treeSummary/registry/checklist/dashboard"));
        archive.put("payload", receipt.getOrDefault("payload", "checklistDashboardReport"));
        archive.put("status", receipt.getOrDefault("status", "WARN"));
        archive.put("verdict", receipt.getOrDefault("verdict", "review"));
        archive.put("section", receipt.getOrDefault("section", "registryChecklist"));
        archive.put("receiptSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci");
        archive.put("deliverySelector", receipt.getOrDefault("deliverySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci"));
        archive.put("bundleSelector", receipt.getOrDefault("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci"));
        archive.put("breadcrumb", receipt.getOrDefault("breadcrumb", ""));
        return archive;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger(String resultFile, Map<String, Object> result) {
        Map<String, Object> ledger = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerSummary(resultFile));
        ledger.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        ledger.put("resultFile", resultFile);
        ledger.put("archive", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveSummary(resultFile)));
        return ledger;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerSummary(String resultFile) {
        Map<String, Object> archive = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchiveSummary(resultFile);
        Map<String, Object> ledger = new LinkedHashMap<>();
        ledger.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger");
        ledger.put("current", archive.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        ledger.put("ledgerVersion", "checklist-dashboard-ledger/v1");
        ledger.put("archiveVersion", archive.getOrDefault("archiveVersion", "checklist-dashboard-archive/v1"));
        ledger.put("entryType", "checklistDashboardArchive");
        ledger.put("recordState", archive.getOrDefault("recordState", "archived"));
        ledger.put("payload", archive.getOrDefault("payload", "checklistDashboardReport"));
        ledger.put("status", archive.getOrDefault("status", "WARN"));
        ledger.put("verdict", archive.getOrDefault("verdict", "review"));
        ledger.put("section", archive.getOrDefault("section", "registryChecklist"));
        ledger.put("archiveSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci");
        ledger.put("receiptSelector", archive.getOrDefault("receiptSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci"));
        ledger.put("deliverySelector", archive.getOrDefault("deliverySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci"));
        ledger.put("bundleSelector", archive.getOrDefault("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardBundle:ci"));
        ledger.put("breadcrumb", archive.getOrDefault("breadcrumb", ""));
        return ledger;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory(String resultFile, Map<String, Object> result) {
        Map<String, Object> history = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistorySummary(resultFile));
        history.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        history.put("resultFile", resultFile);
        history.put("ledger", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerSummary(resultFile)));
        return history;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistorySummary(String resultFile) {
        Map<String, Object> ledger = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedgerSummary(resultFile);
        Map<String, Object> history = new LinkedHashMap<>();
        history.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory");
        history.put("current", ledger.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        history.put("historyVersion", "checklist-dashboard-history/v1");
        history.put("ledgerVersion", ledger.getOrDefault("ledgerVersion", "checklist-dashboard-ledger/v1"));
        history.put("historyDepth", 4);
        history.put("entryType", ledger.getOrDefault("entryType", "checklistDashboardArchive"));
        history.put("recordState", ledger.getOrDefault("recordState", "archived"));
        history.put("payload", ledger.getOrDefault("payload", "checklistDashboardReport"));
        history.put("status", ledger.getOrDefault("status", "WARN"));
        history.put("verdict", ledger.getOrDefault("verdict", "review"));
        history.put("section", ledger.getOrDefault("section", "registryChecklist"));
        history.put("ledgerSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger:ci");
        history.put("archiveSelector", ledger.getOrDefault("archiveSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci"));
        history.put("receiptSelector", ledger.getOrDefault("receiptSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci"));
        history.put("deliverySelector", ledger.getOrDefault("deliverySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci"));
        history.put("breadcrumb", ledger.getOrDefault("breadcrumb", ""));
        return history;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline(String resultFile, Map<String, Object> result) {
        Map<String, Object> timeline = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineSummary(resultFile));
        timeline.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        timeline.put("resultFile", resultFile);
        timeline.put("history", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistoryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistorySummary(resultFile)));
        return timeline;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineSummary(String resultFile) {
        Map<String, Object> history = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistorySummary(resultFile);
        Map<String, Object> timeline = new LinkedHashMap<>();
        timeline.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline");
        timeline.put("current", history.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        timeline.put("timelineVersion", "checklist-dashboard-timeline/v1");
        timeline.put("historyVersion", history.getOrDefault("historyVersion", "checklist-dashboard-history/v1"));
        timeline.put("eventCount", 5);
        timeline.put("events", "delivery>receipt>archive>ledger>history");
        timeline.put("payload", history.getOrDefault("payload", "checklistDashboardReport"));
        timeline.put("status", history.getOrDefault("status", "WARN"));
        timeline.put("verdict", history.getOrDefault("verdict", "review"));
        timeline.put("section", history.getOrDefault("section", "registryChecklist"));
        timeline.put("historySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory:ci");
        timeline.put("ledgerSelector", history.getOrDefault("ledgerSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger:ci"));
        timeline.put("archiveSelector", history.getOrDefault("archiveSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci"));
        timeline.put("receiptSelector", history.getOrDefault("receiptSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardReceipt:ci"));
        timeline.put("deliverySelector", history.getOrDefault("deliverySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDelivery:ci"));
        timeline.put("breadcrumb", history.getOrDefault("breadcrumb", ""));
        return timeline;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace(String resultFile, Map<String, Object> result) {
        Map<String, Object> trace = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceSummary(resultFile));
        trace.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        trace.put("resultFile", resultFile);
        trace.put("timeline", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineSummary(resultFile)));
        return trace;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceSummary(String resultFile) {
        Map<String, Object> timeline = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimelineSummary(resultFile);
        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace");
        trace.put("current", timeline.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        trace.put("traceVersion", "checklist-dashboard-trace/v1");
        trace.put("timelineVersion", timeline.getOrDefault("timelineVersion", "checklist-dashboard-timeline/v1"));
        trace.put("spanCount", timeline.getOrDefault("eventCount", 5));
        trace.put("tracePath", timeline.getOrDefault("events", "delivery>receipt>archive>ledger>history"));
        trace.put("payload", timeline.getOrDefault("payload", "checklistDashboardReport"));
        trace.put("status", timeline.getOrDefault("status", "WARN"));
        trace.put("verdict", timeline.getOrDefault("verdict", "review"));
        trace.put("section", timeline.getOrDefault("section", "registryChecklist"));
        trace.put("timelineSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline:ci");
        trace.put("historySelector", timeline.getOrDefault("historySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory:ci"));
        trace.put("ledgerSelector", timeline.getOrDefault("ledgerSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardLedger:ci"));
        trace.put("archiveSelector", timeline.getOrDefault("archiveSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardArchive:ci"));
        trace.put("breadcrumb", timeline.getOrDefault("breadcrumb", ""));
        return trace;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry(String resultFile, Map<String, Object> result) {
        Map<String, Object> telemetry = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetrySummary(resultFile));
        telemetry.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        telemetry.put("resultFile", resultFile);
        telemetry.put("trace", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceSummary(resultFile)));
        return telemetry;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetrySummary(String resultFile) {
        Map<String, Object> trace = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTraceSummary(resultFile);
        Map<String, Object> telemetry = new LinkedHashMap<>();
        telemetry.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry");
        telemetry.put("current", trace.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        telemetry.put("telemetryVersion", "checklist-dashboard-telemetry/v1");
        telemetry.put("traceVersion", trace.getOrDefault("traceVersion", "checklist-dashboard-trace/v1"));
        telemetry.put("signal", "trace");
        telemetry.put("healthSignal", trace.getOrDefault("status", "WARN"));
        telemetry.put("spanCount", trace.getOrDefault("spanCount", 5));
        telemetry.put("tracePath", trace.getOrDefault("tracePath", "delivery>receipt>archive>ledger>history"));
        telemetry.put("payload", trace.getOrDefault("payload", "checklistDashboardReport"));
        telemetry.put("status", trace.getOrDefault("status", "WARN"));
        telemetry.put("verdict", trace.getOrDefault("verdict", "review"));
        telemetry.put("section", trace.getOrDefault("section", "registryChecklist"));
        telemetry.put("traceSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace:ci");
        telemetry.put("timelineSelector", trace.getOrDefault("timelineSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline:ci"));
        telemetry.put("historySelector", trace.getOrDefault("historySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardHistory:ci"));
        telemetry.put("breadcrumb", trace.getOrDefault("breadcrumb", ""));
        return telemetry;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation(String resultFile, Map<String, Object> result) {
        Map<String, Object> observation = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationSummary(resultFile));
        observation.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        observation.put("resultFile", resultFile);
        observation.put("telemetry", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetrySummary(resultFile)));
        return observation;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationSummary(String resultFile) {
        Map<String, Object> telemetry = resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetrySummary(resultFile);
        Map<String, Object> observation = new LinkedHashMap<>();
        observation.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation");
        observation.put("current", telemetry.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        observation.put("observationVersion", "checklist-dashboard-observation/v1");
        observation.put("telemetryVersion", telemetry.getOrDefault("telemetryVersion", "checklist-dashboard-telemetry/v1"));
        observation.put("observedState", telemetry.getOrDefault("healthSignal", "WARN"));
        observation.put("signal", telemetry.getOrDefault("signal", "trace"));
        observation.put("spanCount", telemetry.getOrDefault("spanCount", 5));
        observation.put("tracePath", telemetry.getOrDefault("tracePath", "delivery>receipt>archive>ledger>history"));
        observation.put("payload", telemetry.getOrDefault("payload", "checklistDashboardReport"));
        observation.put("status", telemetry.getOrDefault("status", "WARN"));
        observation.put("verdict", telemetry.getOrDefault("verdict", "review"));
        observation.put("section", telemetry.getOrDefault("section", "registryChecklist"));
        observation.put("telemetrySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry:ci");
        observation.put("traceSelector", telemetry.getOrDefault("traceSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTrace:ci"));
        observation.put("timelineSelector", telemetry.getOrDefault("timelineSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTimeline:ci"));
        observation.put("breadcrumb", telemetry.getOrDefault("breadcrumb", ""));
        return observation;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal(String resultFile, Map<String, Object> result) {
        Map<String, Object> signal = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(resultFile, result));
        signal.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        signal.put("resultFile", resultFile);
        signal.put("observation", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationSummary(resultFile)));
        return signal;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationCi");
        Map<String, Object> observation = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservationSummary(resultFile);
        Map<String, Object> signal = new LinkedHashMap<>();
        signal.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal");
        signal.put("current", observation.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        signal.put("signalVersion", "checklist-dashboard-signal/v1");
        signal.put("observationVersion", observation.getOrDefault("observationVersion", "checklist-dashboard-observation/v1"));
        signal.put("observedState", observation.getOrDefault("observedState", "WARN"));
        signal.put("status", observation.getOrDefault("status", "WARN"));
        signal.put("verdict", observation.getOrDefault("verdict", "review"));
        signal.put("action", "pass".equals(String.valueOf(observation.getOrDefault("verdict", "review"))) ? "continue" : "review");
        signal.put("payload", observation.getOrDefault("payload", "checklistDashboardReport"));
        signal.put("section", observation.getOrDefault("section", "registryChecklist"));
        signal.put("observationSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation:ci");
        signal.put("telemetrySelector", observation.getOrDefault("telemetrySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardTelemetry:ci"));
        signal.put("breadcrumb", observation.getOrDefault("breadcrumb", ""));
        return signal;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight(String resultFile, Map<String, Object> result) {
        Map<String, Object> insight = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(resultFile, result));
        insight.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        insight.put("resultFile", resultFile);
        insight.put("signal", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(resultFile, result)));
        return insight;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalCi");
        Map<String, Object> signal = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignalSummary(resultFile, result);
        String action = String.valueOf(signal.getOrDefault("action", "review"));
        String observedState = String.valueOf(signal.getOrDefault("observedState", "WARN"));
        String verdict = String.valueOf(signal.getOrDefault("verdict", "review"));
        Map<String, Object> insight = new LinkedHashMap<>();
        insight.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight");
        insight.put("current", signal.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        insight.put("insightVersion", "checklist-dashboard-insight/v1");
        insight.put("signalVersion", signal.getOrDefault("signalVersion", "checklist-dashboard-signal/v1"));
        insight.put("headline", "continue".equals(action) ? "Checklist dashboard signal is clear" : "Checklist dashboard signal needs review");
        insight.put("detail", "Observed state " + observedState + " produced verdict " + verdict + " for " + signal.getOrDefault("payload", "checklistDashboardReport") + ".");
        insight.put("recommendedAction", action);
        insight.put("observedState", observedState);
        insight.put("verdict", verdict);
        insight.put("section", signal.getOrDefault("section", "registryChecklist"));
        insight.put("signalSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal:ci");
        insight.put("observationSelector", signal.getOrDefault("observationSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardObservation:ci"));
        insight.put("breadcrumb", signal.getOrDefault("breadcrumb", ""));
        return insight;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation(String resultFile, Map<String, Object> result) {
        Map<String, Object> recommendation = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(resultFile, result));
        recommendation.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        recommendation.put("resultFile", resultFile);
        recommendation.put("insight", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(resultFile, result)));
        return recommendation;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightCi");
        Map<String, Object> insight = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsightSummary(resultFile, result);
        String action = String.valueOf(insight.getOrDefault("recommendedAction", "review"));
        String priority = "continue".equals(action) ? "low" : "high";
        Map<String, Object> recommendation = new LinkedHashMap<>();
        recommendation.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation");
        recommendation.put("current", insight.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        recommendation.put("recommendationVersion", "checklist-dashboard-recommendation/v1");
        recommendation.put("insightVersion", insight.getOrDefault("insightVersion", "checklist-dashboard-insight/v1"));
        recommendation.put("action", action);
        recommendation.put("priority", priority);
        recommendation.put("owner", "registryChecklist");
        recommendation.put("reason", insight.getOrDefault("detail", "Review checklist dashboard signal."));
        recommendation.put("headline", insight.getOrDefault("headline", "Checklist dashboard signal needs review"));
        recommendation.put("observedState", insight.getOrDefault("observedState", "WARN"));
        recommendation.put("verdict", insight.getOrDefault("verdict", "review"));
        recommendation.put("insightSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight:ci");
        recommendation.put("signalSelector", insight.getOrDefault("signalSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardSignal:ci"));
        recommendation.put("breadcrumb", insight.getOrDefault("breadcrumb", ""));
        return recommendation;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision(String resultFile, Map<String, Object> result) {
        Map<String, Object> decision = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(resultFile, result));
        decision.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        decision.put("resultFile", resultFile);
        decision.put("recommendation", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(resultFile, result)));
        return decision;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationCi");
        Map<String, Object> recommendation = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendationSummary(resultFile, result);
        String action = String.valueOf(recommendation.getOrDefault("action", "review"));
        boolean allow = "continue".equals(action);
        Map<String, Object> decision = new LinkedHashMap<>();
        decision.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision");
        decision.put("current", recommendation.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        decision.put("decisionVersion", "checklist-dashboard-decision/v1");
        decision.put("recommendationVersion", recommendation.getOrDefault("recommendationVersion", "checklist-dashboard-recommendation/v1"));
        decision.put("allow", allow);
        decision.put("decision", allow ? "allow" : "block");
        decision.put("blockingReason", allow ? "" : recommendation.getOrDefault("reason", "Review checklist dashboard signal."));
        decision.put("action", action);
        decision.put("priority", recommendation.getOrDefault("priority", allow ? "low" : "high"));
        decision.put("owner", recommendation.getOrDefault("owner", "registryChecklist"));
        decision.put("recommendationSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardRecommendation:ci");
        decision.put("insightSelector", recommendation.getOrDefault("insightSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardInsight:ci"));
        decision.put("breadcrumb", recommendation.getOrDefault("breadcrumb", ""));
        return decision;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate(String resultFile, Map<String, Object> result) {
        Map<String, Object> gate = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(resultFile, result));
        gate.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        gate.put("resultFile", resultFile);
        gate.put("decision", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(resultFile, result)));
        return gate;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionCi");
        Map<String, Object> decision = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecisionSummary(resultFile, result);
        boolean allow = Boolean.TRUE.equals(decision.get("allow"));
        Map<String, Object> gate = new LinkedHashMap<>();
        gate.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate");
        gate.put("gateVersion", "checklist-dashboard-gate/v1");
        gate.put("decisionVersion", decision.getOrDefault("decisionVersion", "checklist-dashboard-decision/v1"));
        gate.put("gate", allow ? "open" : "closed");
        gate.put("allow", allow);
        gate.put("exitCode", allow ? 0 : 1);
        gate.put("reason", decision.getOrDefault("blockingReason", ""));
        gate.put("decisionSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision:ci");
        gate.put("breadcrumb", decision.getOrDefault("breadcrumb", ""));
        return gate;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine(String resultFile, Map<String, Object> result) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(resultFile, result);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(resultFile, new LinkedHashMap<>());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateCi");
        Map<String, Object> gate = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(resultFile, result);
        return "checklistDashboardGate gate=" + gate.getOrDefault("gate", "closed")
                + " allow=" + gate.getOrDefault("allow", false)
                + " exitCode=" + gate.getOrDefault("exitCode", 1)
                + " reason=" + gate.getOrDefault("reason", "");
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson(String resultFile, Map<String, Object> result) {
        Map<String, Object> gateJson = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(resultFile, result));
        gateJson.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        gateJson.put("resultFile", resultFile);
        return gateJson;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateCi");
        Map<String, Object> gate = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary(resultFile, result);
        Map<String, Object> gateJson = new LinkedHashMap<>();
        gateJson.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson");
        gateJson.put("gateJsonVersion", "checklist-dashboard-gate-json/v1");
        gateJson.put("gate", gate.getOrDefault("gate", "closed"));
        gateJson.put("allow", gate.getOrDefault("allow", false));
        gateJson.put("exitCode", gate.getOrDefault("exitCode", 1));
        gateJson.put("reason", gate.getOrDefault("reason", ""));
        gateJson.put("decisionSelector", gate.getOrDefault("decisionSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardDecision:ci"));
        return gateJson;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv(String resultFile, Map<String, Object> result) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(resultFile, result);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(resultFile, new LinkedHashMap<>());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi");
        Map<String, Object> gateJson = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(resultFile, result);
        return "CHECKLIST_DASHBOARD_GATE=" + gateJson.getOrDefault("gate", "closed")
                + " CHECKLIST_DASHBOARD_ALLOW=" + gateJson.getOrDefault("allow", false)
                + " CHECKLIST_DASHBOARD_EXIT_CODE=" + gateJson.getOrDefault("exitCode", 1)
                + " CHECKLIST_DASHBOARD_REASON=\"" + String.valueOf(gateJson.getOrDefault("reason", "")).replace("\"", "'") + "\"";
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown(String resultFile, Map<String, Object> result) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(resultFile, result);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(resultFile, new LinkedHashMap<>());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi");
        Map<String, Object> gateJson = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(resultFile, result);
        return "**Checklist dashboard gate:** `" + gateJson.getOrDefault("gate", "closed") + "`"
                + " | allow=`" + gateJson.getOrDefault("allow", false) + "`"
                + " | exitCode=`" + gateJson.getOrDefault("exitCode", 1) + "`"
                + " | reason: " + gateJson.getOrDefault("reason", "");
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundle(String resultFile, Map<String, Object> result) {
        Map<String, Object> gateSummary = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(resultFile, result));
        gateSummary.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        gateSummary.put("resultFile", resultFile);
        return gateSummary;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> gateSummary = new LinkedHashMap<>();
        gateSummary.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary");
        gateSummary.put("gateSummaryVersion", "checklist-dashboard-gate-summary/v1");
        gateSummary.put("json", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(resultFile, result)));
        gateSummary.put("line", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(resultFile, result)));
        gateSummary.put("env", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(resultFile, result)));
        gateSummary.put("markdown", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(resultFile, result)));
        gateSummary.put("gateJsonSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson:ci");
        return gateSummary;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact(String resultFile, Map<String, Object> result) {
        Map<String, Object> gateArtifact = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(resultFile, result));
        gateArtifact.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        return gateArtifact;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> gateArtifact = new LinkedHashMap<>();
        gateArtifact.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact");
        gateArtifact.put("gateArtifactVersion", "checklist-dashboard-gate-artifact/v1");
        gateArtifact.put("artifactKind", "checklist-dashboard-gate");
        gateArtifact.put("resultFile", resultFile);
        gateArtifact.put("gateSummary", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(resultFile, result)));
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("gate", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGate:ci");
        selectors.put("json", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson:ci");
        selectors.put("line", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine:ci");
        selectors.put("env", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv:ci");
        selectors.put("markdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown:ci");
        selectors.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary:ci");
        gateArtifact.put("selectors", selectors);
        return gateArtifact;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest(String resultFile, Map<String, Object> result) {
        Map<String, Object> uploadManifest = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile, result));
        uploadManifest.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        return uploadManifest;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi");
        Map<String, Object> artifact = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(resultFile, result);
        Map<String, Object> uploadManifest = new LinkedHashMap<>();
        uploadManifest.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest");
        uploadManifest.put("uploadManifestVersion", "checklist-dashboard-gate-upload-manifest/v1");
        uploadManifest.put("artifactKind", artifact.getOrDefault("artifactKind", "checklist-dashboard-gate"));
        uploadManifest.put("artifactLabel", "Checklist dashboard gate");
        uploadManifest.put("resultFile", artifact.getOrDefault("resultFile", resultFile));
        uploadManifest.put("artifactSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci");
        uploadManifest.put("artifactName", "checklist-dashboard-gate");
        uploadManifest.put("files", checklistDashboardGateUploadFiles());
        return uploadManifest;
    }

    private static List<Map<String, Object>> checklistDashboardGateUploadFiles() {
        return List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                uploadManifestFile("artifact", "checklist-dashboard-gate-artifact.json", "application/json", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci"),
                uploadManifestFile("summary", "checklist-dashboard-gate-summary.json", "application/json", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary:ci"),
                uploadManifestFile("json", "checklist-dashboard-gate.json", "application/json", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson:ci"),
                uploadManifestFile("line", "checklist-dashboard-gate.txt", "text/plain", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine:ci"),
                uploadManifestFile("env", "checklist-dashboard-gate.env", "text/plain", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv:ci"),
                uploadManifestFile("markdown", "checklist-dashboard-gate.md", "text/markdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown:ci")});
    }

    private static Map<String, Object> uploadManifestFile(String kind, String filename, String contentType, String selector) {
        Map<String, Object> file = new LinkedHashMap<>();
        file.put("kind", kind);
        file.put("filename", filename);
        file.put("contentType", contentType);
        file.put("selector", selector);
        file.put("label", "Checklist dashboard gate " + kind);
        return file;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript(String resultFile, Map<String, Object> result) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptSummary(resultFile, result);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptSummary(resultFile, new LinkedHashMap<>());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi");
        Map<String, Object> manifest = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile, result);
        Object filesObject = manifest.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : checklistDashboardGateUploadFiles();
        StringBuilder script = new StringBuilder();
        script.append("#!/usr/bin/env sh\n");
        script.append("set -eu\n");
        script.append("RESULT_FILE=${RESULT_FILE:-").append(shellQuote(resultFile)).append("}\n");
        script.append("OUT_DIR=${OUT_DIR:-./checklist-dashboard-gate}\n");
        script.append("mkdir -p \"$OUT_DIR\"\n");
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String selector = String.valueOf(file.getOrDefault("selector", ""));
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            String format = "application/json".equals(file.getOrDefault("contentType", "")) ? "json" : "text";
            if (selector.isBlank() || filename.isBlank()) {
                continue;
            }
            script.append("jbang gollek/examples/jbang/trainer/trainer_byte_latent_train_infer_compare_inspector.java \"$RESULT_FILE\" ")
                    .append(shellQuote(selector))
                    .append(" ")
                    .append(format)
                    .append(" > \"$OUT_DIR/")
                    .append(filename.replace("\"", ""))
                    .append("\"\n");
        }
        return script.toString().trim();
    }

    private static String shellQuote(String value) {
        return "'" + String.valueOf(value).replace("'", "'\"'\"'") + "'";
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme(String resultFile, Map<String, Object> result) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeSummary(resultFile, result);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeSummary(resultFile, new LinkedHashMap<>());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi");
        Map<String, Object> manifest = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile, result);
        Object filesObject = manifest.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : checklistDashboardGateUploadFiles();
        StringBuilder readme = new StringBuilder();
        readme.append("# Checklist Dashboard Gate Artifacts\n\n");
        readme.append("Artifact: `").append(manifest.getOrDefault("artifactName", "checklist-dashboard-gate")).append("`\n\n");
        readme.append("Source result file: `").append(manifest.getOrDefault("resultFile", resultFile)).append("`\n\n");
        readme.append("Artifact selector: `").append(manifest.getOrDefault("artifactSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci")).append("`\n\n");
        readme.append("| File | Kind | Content type | Selector |\n");
        readme.append("| --- | --- | --- | --- |\n");
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            readme.append("| `").append(file.getOrDefault("filename", "")).append("` | `")
                    .append(file.getOrDefault("kind", "")).append("` | `")
                    .append(file.getOrDefault("contentType", "")).append("` | `")
                    .append(file.getOrDefault("selector", "")).append("` |\n");
        }
        readme.append("\nGenerate these files with `resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript:ci`.\n");
        return readme.toString().trim();
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle(String resultFile, Map<String, Object> result) {
        Map<String, Object> uploadBundle = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleSummary(resultFile, result));
        uploadBundle.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        return uploadBundle;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundleSummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> uploadBundle = new LinkedHashMap<>();
        uploadBundle.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle");
        uploadBundle.put("uploadBundleVersion", "checklist-dashboard-gate-upload-bundle/v1");
        uploadBundle.put("artifact", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(resultFile, result)));
        uploadBundle.put("manifest", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile, result)));
        uploadBundle.put("script", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScriptSummary(resultFile, result)));
        uploadBundle.put("readme", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadmeSummary(resultFile, result)));
        uploadBundle.put("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci");
        uploadBundle.put("scriptSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript:ci");
        uploadBundle.put("readmeSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme:ci");
        return uploadBundle;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex(String resultFile, Map<String, Object> result) {
        Map<String, Object> uploadIndex = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexSummary(resultFile, result));
        uploadIndex.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        uploadIndex.put("resultFile", resultFile);
        return uploadIndex;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndexSummary(String resultFile, Map<String, Object> result) {
        List<Map<String, Object>> selectors = checklistDashboardGateUploadIndexRows(resultFile);
        Map<String, Object> uploadIndex = new LinkedHashMap<>();
        uploadIndex.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex");
        uploadIndex.put("uploadIndexVersion", "checklist-dashboard-gate-upload-index/v1");
        uploadIndex.put("selectorCount", selectors.size());
        uploadIndex.put("selectors", selectors);
        uploadIndex.put("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci");
        return uploadIndex;
    }

    private static List<Map<String, Object>> checklistDashboardGateUploadIndexRows(String resultFile) {
        return List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                gateUploadIndexRow("artifact", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci", resultFile, "Gate artifact payload with summary and selector metadata"),
                gateUploadIndexRow("manifest", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci", resultFile, "Upload filenames, labels, content types, and source selectors"),
                gateUploadIndexRow("script", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript:ci", resultFile, "Shell script that materializes upload files from selectors"),
                gateUploadIndexRow("readme", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadReadme:ci", resultFile, "Human-readable README for the upload bundle"),
                gateUploadIndexRow("filesIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci", resultFile, "Compact upload filename, content type, selector, and command rows"),
                gateUploadIndexRow("filesLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine:ci", resultFile, "One-line upload filename summary for logs and notifications"),
                gateUploadIndexRow("filesJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine:ci", resultFile, "JSONL upload filename summary for structured logs"),
                gateUploadIndexRow("filesCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine:ci", resultFile, "CSV upload filename summary for spreadsheets and archives"),
                gateUploadIndexRow("filesTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine:ci", resultFile, "TSV upload filename summary for spreadsheets and archives"),
                gateUploadIndexRow("filesEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv:ci", resultFile, "Shell environment upload filename summary for CI scripts"),
                gateUploadIndexRow("filesMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown:ci", resultFile, "Markdown upload filename table for release comments and review pages"),
                gateUploadIndexRow("filesChecksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci", resultFile, "SHA-256 upload file checksums for archive integrity checks"),
                gateUploadIndexRow("filesChecksumsSummary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci", resultFile, "Compact SHA-256 upload checksum dashboard summary"),
                gateUploadIndexRow("filesChecksumsBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge:ci", resultFile, "Tiny SHA-256 upload checksum badge label"),
                gateUploadIndexRow("filesChecksumsCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard:ci", resultFile, "Small SHA-256 upload checksum UI card"),
                gateUploadIndexRow("filesChecksumsPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci", resultFile, "Richer SHA-256 upload checksum dashboard panel"),
                gateUploadIndexRow("filesChecksumsWidget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci", resultFile, "Compact SHA-256 upload checksum dashboard widget"),
                gateUploadIndexRow("filesChecksumsTile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci", resultFile, "Tiny SHA-256 upload checksum dashboard tile"),
                gateUploadIndexRow("filesChecksumsMetric", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric:ci", resultFile, "Numeric SHA-256 upload checksum dashboard metric"),
                gateUploadIndexRow("filesChecksumsTrend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci", resultFile, "Trend-ready SHA-256 upload checksum dashboard metric"),
                gateUploadIndexRow("filesChecksumsGauge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci", resultFile, "Gauge-ready SHA-256 upload checksum dashboard status"),
                gateUploadIndexRow("filesChecksumsSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci", resultFile, "Summary-card SHA-256 upload checksum dashboard payload"),
                gateUploadIndexRow("filesChecksumsActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar:ci", resultFile, "Action-bar SHA-256 upload checksum dashboard shortcuts"),
                gateUploadIndexRow("filesChecksumsMenu", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu:ci", resultFile, "Grouped SHA-256 upload checksum dashboard menu"),
                gateUploadIndexRow("filesChecksumsToolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar:ci", resultFile, "Toolbar SHA-256 upload checksum dashboard controls"),
                gateUploadIndexRow("filesChecksumsCommandPalette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette:ci", resultFile, "Command-palette SHA-256 upload checksum navigation"),
                gateUploadIndexRow("filesChecksumsSearchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex:ci", resultFile, "Search-index SHA-256 upload checksum selector entries"),
                gateUploadIndexRow("filesChecksumsSearchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci", resultFile, "Sample search result SHA-256 upload checksum selector entries"),
                gateUploadIndexRow("filesChecksumsFilterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci", resultFile, "Filter-state SHA-256 upload checksum search defaults"),
                gateUploadIndexRow("filesChecksumsViewState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci", resultFile, "View-state SHA-256 upload checksum dashboard payload"),
                gateUploadIndexRow("filesChecksumsEmptyState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState:ci", resultFile, "Empty-state SHA-256 upload checksum dashboard fallback"),
                gateUploadIndexRow("filesChecksumsLoadingState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci", resultFile, "Loading-state SHA-256 upload checksum dashboard fallback"),
                gateUploadIndexRow("filesChecksumsErrorState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci", resultFile, "Error-state SHA-256 upload checksum dashboard recovery"),
                gateUploadIndexRow("filesChecksumsToastState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci", resultFile, "Toast-state SHA-256 upload checksum notifications"),
                gateUploadIndexRow("filesChecksumsNotificationCenter", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci", resultFile, "Notification-center SHA-256 upload checksum history"),
                gateUploadIndexRow("filesChecksumsNotificationBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci", resultFile, "Notification-badge SHA-256 upload checksum counts"),
                gateUploadIndexRow("filesChecksumsNotificationPanel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci", resultFile, "Notification-panel SHA-256 upload checksum summary"),
                gateUploadIndexRow("filesChecksumsNotificationPanelLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine:ci", resultFile, "One-line notification-panel SHA-256 upload checksum summary"),
                gateUploadIndexRow("filesChecksumsNotificationPanelJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine:ci", resultFile, "JSONL notification-panel SHA-256 upload checksum summary"),
                gateUploadIndexRow("filesChecksumsNotificationPanelCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine:ci", resultFile, "CSV notification-panel SHA-256 upload checksum summary"),
                gateUploadIndexRow("filesChecksumsNotificationPanelTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine:ci", resultFile, "TSV notification-panel SHA-256 upload checksum summary"),
                gateUploadIndexRow("filesChecksumsNotificationPanelMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci", resultFile, "Markdown notification-panel SHA-256 upload checksum summary"),
                gateUploadIndexRow("filesChecksumsNotificationPanelEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci", resultFile, "Shell environment notification-panel SHA-256 upload checksum summary"),
                gateUploadIndexRow("filesChecksumsNotificationPanelSummaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard:ci", resultFile, "Summary-card notification-panel SHA-256 upload checksum payload"),
                gateUploadIndexRow("filesChecksumsNotificationPanelActionBar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar:ci", resultFile, "Action-bar notification-panel SHA-256 upload checksum shortcuts"),
                gateUploadIndexRow("filesChecksumsNotificationPanelMenu", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu:ci", resultFile, "Menu notification-panel SHA-256 upload checksum navigation"),
                gateUploadIndexRow("filesChecksumsNotificationPanelToolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar:ci", resultFile, "Toolbar notification-panel SHA-256 upload checksum controls"),
                gateUploadIndexRow("filesChecksumsNotificationPanelCommandPalette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci", resultFile, "Command-palette notification-panel SHA-256 upload checksum commands"),
                gateUploadIndexRow("filesChecksumsNotificationPanelSearchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci", resultFile, "Search-index notification-panel SHA-256 upload checksum commands"),
                gateUploadIndexRow("filesChecksumsNotificationPanelSearchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci", resultFile, "Search-results notification-panel SHA-256 upload checksum commands"),
                gateUploadIndexRow("filesChecksumsNotificationPanelFilterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci", resultFile, "Filter-state notification-panel SHA-256 upload checksum commands"),
                gateUploadIndexRow("filesChecksumsNotificationPanelViewState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci", resultFile, "View-state notification-panel SHA-256 upload checksum payload"),
                gateUploadIndexRow("filesChecksumsNotificationPanelEmptyState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState:ci", resultFile, "Empty-state notification-panel SHA-256 upload checksum fallback"),
                gateUploadIndexRow("filesChecksumsNotificationPanelLoadingState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci", resultFile, "Loading-state notification-panel SHA-256 upload checksum fallback"),
                gateUploadIndexRow("filesChecksumsNotificationPanelErrorState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState:ci", resultFile, "Error-state notification-panel SHA-256 upload checksum recovery"),
                gateUploadIndexRow("filesChecksumsNotificationPanelToastState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState:ci", resultFile, "Toast-state notification-panel SHA-256 upload checksum notifications"),
                gateUploadIndexRow("filesChecksumsNotificationPanelNotificationCenter", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter:ci", resultFile, "Notification-center notification-panel SHA-256 upload checksum history"),
                gateUploadIndexRow("filesChecksumsNotificationPanelNotificationBadge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge:ci", resultFile, "Notification-badge notification-panel SHA-256 upload checksum counts"),
                gateUploadIndexRow("filesChecksumsNotificationPanelNotificationLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine:ci", resultFile, "One-line notification-panel badge state for CI logs"),
                gateUploadIndexRow("filesChecksumsNotificationPanelNotificationJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine:ci", resultFile, "JSONL notification-panel badge state for structured CI logs"),
                gateUploadIndexRow("filesChecksumsLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci", resultFile, "One-line SHA-256 upload checksum summary for CI logs"),
                gateUploadIndexRow("filesChecksumsJsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci", resultFile, "JSONL SHA-256 upload checksum summary for structured logs"),
                gateUploadIndexRow("filesChecksumsCsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine:ci", resultFile, "CSV SHA-256 upload checksum summary for spreadsheets and archives"),
                gateUploadIndexRow("filesChecksumsTsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine:ci", resultFile, "TSV SHA-256 upload checksum summary for tab-safe reports"),
                gateUploadIndexRow("filesChecksumsMarkdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci", resultFile, "Markdown SHA-256 upload checksum table for release comments"),
                gateUploadIndexRow("filesChecksumsEnv", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv:ci", resultFile, "Shell environment SHA-256 upload checksum summary for CI scripts"),
                gateUploadIndexRow("bundle", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci", resultFile, "One-shot upload payload with artifact, manifest, script, and README")});
    }

    private static Map<String, Object> gateUploadIndexRow(String name, String selector, String resultFile, String description) {
        Map<String, Object> row = resultAliasRow(name, selector, resultFile);
        row.put("description", description);
        return row;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex(String resultFile, Map<String, Object> result) {
        Map<String, Object> filesIndex = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result));
        filesIndex.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        filesIndex.put("resultFile", resultFile);
        return filesIndex;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi");
        Map<String, Object> manifest = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile, result);
        Object filesObject = manifest.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : checklistDashboardGateUploadFiles();
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            rows.add(gateUploadFileIndexRow(asMap((Map<?, ?>) fileObject), resultFile));
        }
        Map<String, Object> filesIndex = new LinkedHashMap<>();
        filesIndex.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex");
        filesIndex.put("uploadFilesIndexVersion", "checklist-dashboard-gate-upload-files-index/v1");
        filesIndex.put("fileCount", rows.size());
        filesIndex.put("files", List.copyOf(rows));
        filesIndex.put("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci");
        filesIndex.put("scriptSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadScript:ci");
        return filesIndex;
    }

    private static Map<String, Object> gateUploadFileIndexRow(Map<String, Object> file, String resultFile) {
        String selector = String.valueOf(file.getOrDefault("selector", ""));
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("filename", file.getOrDefault("filename", ""));
        row.put("kind", file.getOrDefault("kind", ""));
        row.put("contentType", file.getOrDefault("contentType", ""));
        row.put("selector", selector);
        row.put("command", "jbang trainer/trainer_byte_latent_train_infer_compare_inspector.java " + resultFile + " " + selector);
        return row;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        Map<String, Object> filesIndex = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result);
        Object filesObject = filesIndex.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> filenames = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            if (!filename.isBlank()) {
                filenames.add(filename);
            }
        }
        return "checklistDashboardGateUploadFiles fileCount=" + filenames.size()
                + " files=" + (filenames.isEmpty() ? "none" : String.join(",", filenames));
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        Map<String, Object> filesIndex = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result);
        Object filesObject = filesIndex.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> filenames = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            if (!filename.isBlank()) {
                filenames.add(filename);
            }
        }
        Map<String, Object> jsonLine = new LinkedHashMap<>();
        jsonLine.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesJsonLine");
        jsonLine.put("uploadFilesJsonLineVersion", "checklist-dashboard-gate-upload-files-json-line/v1");
        jsonLine.put("fileCount", filenames.size());
        jsonLine.put("filenames", List.copyOf(filenames));
        jsonLine.put("filesIndexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci");
        try {
            return JSON_LINE.writeValueAsString(jsonLine);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to render gate upload files JSON line.", exception);
        }
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        Map<String, Object> filesIndex = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result);
        Object filesObject = filesIndex.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> filenames = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            if (!filename.isBlank()) {
                filenames.add(filename);
            }
        }
        List<String> headers = List.of("summary", "csvLineVersion", "fileCount", "files", "filesIndexSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesCsvLine",
                "checklist-dashboard-gate-upload-files-csv-line/v1",
                String.valueOf(filenames.size()),
                String.join(",", filenames),
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci");
        return String.join(",", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList()) + "\n"
                + String.join(",", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        Map<String, Object> filesIndex = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result);
        Object filesObject = filesIndex.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> filenames = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            if (!filename.isBlank()) {
                filenames.add(filename);
            }
        }
        List<String> headers = List.of("summary", "tsvLineVersion", "fileCount", "files", "filesIndexSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesTsvLine",
                "checklist-dashboard-gate-upload-files-tsv-line/v1",
                String.valueOf(filenames.size()),
                String.join(",", filenames),
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci");
        return String.join("\t", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList()) + "\n"
                + String.join("\t", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesEnv(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        Map<String, Object> filesIndex = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result);
        Object filesObject = filesIndex.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> filenames = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            if (!filename.isBlank()) {
                filenames.add(filename);
            }
        }
        List<String> lines = new ArrayList<>();
        lines.add("GATE_UPLOAD_FILES_ENV_VERSION=" + shellQuote("checklist-dashboard-gate-upload-files-env/v1"));
        lines.add("GATE_UPLOAD_FILE_COUNT=" + filenames.size());
        lines.add("GATE_UPLOAD_FILES=" + shellQuote(String.join(",", filenames)));
        lines.add("GATE_UPLOAD_FILES_INDEX_SELECTOR=" + shellQuote("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci"));
        for (int index = 0; index < filenames.size(); index++) {
            lines.add("GATE_UPLOAD_FILE_" + (index + 1) + "=" + shellQuote(filenames.get(index)));
        }
        return String.join("\n", lines);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesMarkdown(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        Map<String, Object> filesIndex = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result);
        Object filesObject = filesIndex.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> lines = new ArrayList<>();
        lines.add("**Checklist dashboard gate upload files**");
        lines.add("");
        lines.add("| Filename | Kind | Content type | Selector |");
        lines.add("| --- | --- | --- | --- |");
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            lines.add("| " + escapeMarkdownTableCell(String.valueOf(file.getOrDefault("filename", "")))
                    + " | " + escapeMarkdownTableCell(String.valueOf(file.getOrDefault("kind", "")))
                    + " | " + escapeMarkdownTableCell(String.valueOf(file.getOrDefault("contentType", "")))
                    + " | `" + escapeMarkdownTableCode(String.valueOf(file.getOrDefault("selector", ""))) + "` |");
        }
        lines.add("");
        lines.add("Files index selector: `resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci`");
        return String.join("\n", lines);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexCi");
        Map<String, Object> filesIndex = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndexSummary(resultFile, result);
        Object filesObject = filesIndex.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<Map<String, Object>> checksumRows = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String contentType = String.valueOf(file.getOrDefault("contentType", ""));
            String content = renderGateUploadFileContent(resultFile, result, String.valueOf(file.getOrDefault("selector", "")), contentType);
            byte[] bytes = content.getBytes(StandardCharsets.UTF_8);
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("filename", file.getOrDefault("filename", ""));
            row.put("kind", file.getOrDefault("kind", ""));
            row.put("contentType", contentType);
            row.put("selector", file.getOrDefault("selector", ""));
            row.put("byteLength", bytes.length);
            row.put("sha256", sha256Hex(bytes));
            checksumRows.add(row);
        }
        Map<String, Object> checksums = new LinkedHashMap<>();
        checksums.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums");
        checksums.put("uploadFilesChecksumsVersion", "checklist-dashboard-gate-upload-files-checksums/v1");
        checksums.put("algorithm", "SHA-256");
        checksums.put("fileCount", checksumRows.size());
        checksums.put("files", List.copyOf(checksumRows));
        checksums.put("filesIndexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci");
        return checksums;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        Map<String, Object> checksums = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile, result);
        Object filesObject = checksums.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> pairs = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            String sha256 = String.valueOf(file.getOrDefault("sha256", ""));
            if (!filename.isBlank() && !sha256.isBlank()) {
                pairs.add(filename + "=" + sha256);
            }
        }
        return "checklistDashboardGateUploadFilesChecksums algorithm="
                + checksums.getOrDefault("algorithm", "SHA-256")
                + " fileCount=" + checksums.getOrDefault("fileCount", pairs.size())
                + " checksums=" + (pairs.isEmpty() ? "none" : String.join(",", pairs));
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        Map<String, Object> checksums = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile, result);
        Object filesObject = checksums.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> previewFilenames = new ArrayList<>();
        long totalBytes = 0L;
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            if (previewFilenames.size() < 3) {
                String filename = String.valueOf(file.getOrDefault("filename", ""));
                if (!filename.isBlank()) {
                    previewFilenames.add(filename);
                }
            }
            Object byteLength = file.get("byteLength");
            if (byteLength instanceof Number) {
                totalBytes += ((Number) byteLength).longValue();
            }
        }
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary");
        summary.put("uploadFilesChecksumsSummaryVersion", "checklist-dashboard-gate-upload-files-checksums-summary/v1");
        summary.put("algorithm", checksums.getOrDefault("algorithm", "SHA-256"));
        summary.put("fileCount", checksums.getOrDefault("fileCount", files.size()));
        summary.put("totalBytes", totalBytes);
        summary.put("previewFileCount", previewFilenames.size());
        summary.put("previewFilenames", List.copyOf(previewFilenames));
        summary.put("checksumsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        summary.put("checksumsLineSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci");
        return summary;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCi");
        Map<String, Object> summary = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(resultFile, result);
        return "checksums: " + summary.getOrDefault("algorithm", "SHA-256")
                + " " + summary.getOrDefault("fileCount", 0) + " files"
                + " " + summary.getOrDefault("totalBytes", 0) + " bytes";
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCi");
        Map<String, Object> summary = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(resultFile, result);
        Map<String, Object> card = new LinkedHashMap<>();
        card.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard");
        card.put("uploadFilesChecksumsCardVersion", "checklist-dashboard-gate-upload-files-checksums-card/v1");
        card.put("title", "Upload checksums");
        card.put("badge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(resultFile, result));
        card.put("algorithm", summary.getOrDefault("algorithm", "SHA-256"));
        card.put("fileCount", summary.getOrDefault("fileCount", 0));
        card.put("totalBytes", summary.getOrDefault("totalBytes", 0));
        card.put("previewFilenames", summary.getOrDefault("previewFilenames", List.of()));
        card.put("summarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci");
        card.put("checksumsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        return card;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCardCi");
        Map<String, Object> card = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard(resultFile, result);
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("checksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        selectors.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci");
        selectors.put("badge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge:ci");
        selectors.put("card", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCard:ci");
        selectors.put("line", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci");
        selectors.put("jsonLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci");
        selectors.put("csvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine:ci");
        selectors.put("tsvLine", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine:ci");
        selectors.put("markdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci");
        selectors.put("env", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv:ci");
        Map<String, Object> panel = new LinkedHashMap<>();
        panel.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel");
        panel.put("uploadFilesChecksumsPanelVersion", "checklist-dashboard-gate-upload-files-checksums-panel/v1");
        panel.put("title", "Upload checksum integrity");
        panel.put("description", "SHA-256 upload file checksum dashboard panel");
        panel.put("badge", card.getOrDefault("badge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(resultFile, result)));
        panel.put("algorithm", card.getOrDefault("algorithm", "SHA-256"));
        panel.put("fileCount", card.getOrDefault("fileCount", 0));
        panel.put("totalBytes", card.getOrDefault("totalBytes", 0));
        panel.put("previewFilenames", card.getOrDefault("previewFilenames", List.of()));
        panel.put("card", card);
        panel.put("selectors", selectors);
        return panel;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanelCi");
        Map<String, Object> panel = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel(resultFile, result);
        Map<String, Object> links = new LinkedHashMap<>();
        links.put("panel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci");
        links.put("checksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        links.put("line", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci");
        links.put("markdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci");
        Map<String, Object> widget = new LinkedHashMap<>();
        widget.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget");
        widget.put("uploadFilesChecksumsWidgetVersion", "checklist-dashboard-gate-upload-files-checksums-widget/v1");
        widget.put("title", panel.getOrDefault("title", "Upload checksum integrity"));
        widget.put("badge", panel.getOrDefault("badge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(resultFile, result)));
        widget.put("metricLine", panel.getOrDefault("fileCount", 0) + " files | " + panel.getOrDefault("totalBytes", 0) + " bytes | " + panel.getOrDefault("algorithm", "SHA-256"));
        widget.put("fileCount", panel.getOrDefault("fileCount", 0));
        widget.put("totalBytes", panel.getOrDefault("totalBytes", 0));
        widget.put("algorithm", panel.getOrDefault("algorithm", "SHA-256"));
        widget.put("previewFilenames", panel.getOrDefault("previewFilenames", List.of()));
        widget.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci");
        widget.put("links", links);
        return widget;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidgetCi");
        Map<String, Object> widget = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget(resultFile, result);
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("widget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci");
        selectors.put("panel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci");
        selectors.put("checksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        Map<String, Object> tile = new LinkedHashMap<>();
        tile.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile");
        tile.put("uploadFilesChecksumsTileVersion", "checklist-dashboard-gate-upload-files-checksums-tile/v1");
        tile.put("title", "Upload checksums");
        tile.put("badge", widget.getOrDefault("badge", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsBadge(resultFile, result)));
        tile.put("value", widget.getOrDefault("fileCount", 0) + " files");
        tile.put("caption", widget.getOrDefault("totalBytes", 0) + " bytes | " + widget.getOrDefault("algorithm", "SHA-256"));
        tile.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci");
        tile.put("selectors", selectors);
        return tile;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCi");
        Map<String, Object> summary = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary(resultFile, result);
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("tile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci");
        selectors.put("widget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci");
        selectors.put("panel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci");
        selectors.put("checksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        Map<String, Object> metric = new LinkedHashMap<>();
        metric.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric");
        metric.put("uploadFilesChecksumsMetricVersion", "checklist-dashboard-gate-upload-files-checksums-metric/v1");
        metric.put("name", "upload_checksums");
        metric.put("algorithm", summary.getOrDefault("algorithm", "SHA-256"));
        metric.put("fileCount", summary.getOrDefault("fileCount", 0));
        metric.put("totalBytes", summary.getOrDefault("totalBytes", 0));
        metric.put("unit", "files");
        metric.put("value", summary.getOrDefault("fileCount", 0));
        metric.put("bytesValue", summary.getOrDefault("totalBytes", 0));
        metric.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci");
        metric.put("selectors", selectors);
        return metric;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetricCi");
        Map<String, Object> metric = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric(resultFile, result);
        Object currentValue = metric.getOrDefault("value", 0);
        Object currentBytes = metric.getOrDefault("bytesValue", 0);
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("metric", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric:ci");
        selectors.put("tile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci");
        selectors.put("widget", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsWidget:ci");
        selectors.put("panel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci");
        Map<String, Object> trend = new LinkedHashMap<>();
        trend.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend");
        trend.put("uploadFilesChecksumsTrendVersion", "checklist-dashboard-gate-upload-files-checksums-trend/v1");
        trend.put("name", metric.getOrDefault("name", "upload_checksums"));
        trend.put("algorithm", metric.getOrDefault("algorithm", "SHA-256"));
        trend.put("unit", metric.getOrDefault("unit", "files"));
        trend.put("currentValue", currentValue);
        trend.put("baselineValue", currentValue);
        trend.put("delta", 0);
        trend.put("currentBytesValue", currentBytes);
        trend.put("baselineBytesValue", currentBytes);
        trend.put("bytesDelta", 0);
        trend.put("trendState", "flat");
        trend.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric:ci");
        trend.put("selectors", selectors);
        return trend;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrendCi");
        Map<String, Object> trend = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(resultFile, result);
        int currentValue = numberAsInt(trend.getOrDefault("currentValue", 0));
        int maxValue = 10;
        double ratio = Math.min(1.0d, Math.max(0.0d, currentValue / (double) maxValue));
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("trend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci");
        selectors.put("metric", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMetric:ci");
        selectors.put("tile", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTile:ci");
        selectors.put("panel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsPanel:ci");
        Map<String, Object> gauge = new LinkedHashMap<>();
        gauge.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge");
        gauge.put("uploadFilesChecksumsGaugeVersion", "checklist-dashboard-gate-upload-files-checksums-gauge/v1");
        gauge.put("name", trend.getOrDefault("name", "upload_checksums"));
        gauge.put("algorithm", trend.getOrDefault("algorithm", "SHA-256"));
        gauge.put("unit", trend.getOrDefault("unit", "files"));
        gauge.put("currentValue", currentValue);
        gauge.put("maxValue", maxValue);
        gauge.put("ratio", ratio);
        gauge.put("percent", Math.round(ratio * 100.0d));
        gauge.put("gaugeState", ratio >= 1.0d ? "full" : ratio >= 0.8d ? "busy" : "normal");
        gauge.put("label", currentValue + " / " + maxValue + " files");
        gauge.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci");
        gauge.put("selectors", selectors);
        return gauge;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(String resultFile, Map<String, Object> result) {
        Object gaugeResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGaugeCi");
        Map<String, Object> gauge = gaugeResult instanceof Map ? asMap((Map<?, ?>) gaugeResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge(resultFile, result);
        Object trendResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrendCi");
        Map<String, Object> trend = trendResult instanceof Map ? asMap((Map<?, ?>) trendResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend(resultFile, result);
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("gauge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci");
        selectors.put("trend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci");
        selectors.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci");
        selectors.put("line", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci");
        Map<String, Object> card = new LinkedHashMap<>();
        card.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard");
        card.put("uploadFilesChecksumsSummaryCardVersion", "checklist-dashboard-gate-upload-files-checksums-summary-card/v1");
        card.put("title", "Upload checksums");
        card.put("subtitle", gauge.getOrDefault("label", "0 / 10 files"));
        card.put("status", gauge.getOrDefault("gaugeState", "normal"));
        card.put("algorithm", gauge.getOrDefault("algorithm", "SHA-256"));
        card.put("currentValue", gauge.getOrDefault("currentValue", 0));
        card.put("maxValue", gauge.getOrDefault("maxValue", 10));
        card.put("percent", gauge.getOrDefault("percent", 0));
        card.put("trendState", trend.getOrDefault("trendState", "flat"));
        card.put("delta", trend.getOrDefault("delta", 0));
        card.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci");
        card.put("selectors", selectors);
        card.put("gauge", gauge);
        card.put("trend", trend);
        return card;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(String resultFile, Map<String, Object> result) {
        Object cardResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCardCi");
        Map<String, Object> card = cardResult instanceof Map ? asMap((Map<?, ?>) cardResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard(resultFile, result);
        String tone = "normal".equals(String.valueOf(card.getOrDefault("status", "normal"))) ? "neutral" : "attention";
        List<Map<String, Object>> actions = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checklistAction("summaryCard", "Open card", "primary", tone, "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci"),
                checklistAction("checksums", "Raw checksums", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci"),
                checklistAction("jsonLine", "Copy JSONL", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci"),
                checklistAction("markdown", "Copy markdown", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci"),
                checklistAction("bundle", "Open bundle", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci")});
        Map<String, Object> actionBar = new LinkedHashMap<>();
        actionBar.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar");
        actionBar.put("uploadFilesChecksumsActionBarVersion", "checklist-dashboard-gate-upload-files-checksums-action-bar/v1");
        actionBar.put("label", "Upload checksum actions");
        actionBar.put("status", card.getOrDefault("status", "normal"));
        actionBar.put("tone", tone);
        actionBar.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci");
        actionBar.put("actionCount", actions.size());
        actionBar.put("actions", actions);
        actionBar.put("summaryCard", card);
        return actionBar;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(String resultFile, Map<String, Object> result) {
        Object actionBarResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBarCi");
        Map<String, Object> actionBar = actionBarResult instanceof Map ? asMap((Map<?, ?>) actionBarResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar(resultFile, result);
        List<Map<String, Object>> groups = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checksumMenuGroup("overview", "Overview", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                        checksumMenuItem("summaryCard", "Summary card", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci"),
                        checksumMenuItem("actionBar", "Action bar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsActionBar:ci"),
                        checksumMenuItem("gauge", "Gauge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci"),
                        checksumMenuItem("trend", "Trend", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci")})),
                checksumMenuGroup("integrity", "Integrity", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                        checksumMenuItem("checksums", "Raw checksums", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci"),
                        checksumMenuItem("summary", "Checksum summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci"),
                        checksumMenuItem("markdown", "Markdown table", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci")})),
                checksumMenuGroup("logs", "Logs and exports", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                        checksumMenuItem("line", "Text line", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLine:ci"),
                        checksumMenuItem("jsonLine", "JSONL", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci"),
                        checksumMenuItem("csvLine", "CSV", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine:ci"),
                        checksumMenuItem("tsvLine", "TSV", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine:ci"),
                        checksumMenuItem("env", "Environment", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv:ci")})),
                checksumMenuGroup("upload", "Upload handoff", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                        checksumMenuItem("filesIndex", "Files index", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesIndex:ci"),
                        checksumMenuItem("uploadIndex", "Upload index", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci"),
                        checksumMenuItem("bundle", "Upload bundle", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci")}))});
        int itemCount = 0;
        for (Map<String, Object> group : groups) {
            Object items = group.get("items");
            if (items instanceof List) {
                itemCount += ((List<?>) items).size();
            }
        }
        Map<String, Object> menu = new LinkedHashMap<>();
        menu.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu");
        menu.put("uploadFilesChecksumsMenuVersion", "checklist-dashboard-gate-upload-files-checksums-menu/v1");
        menu.put("label", "Upload checksum menu");
        menu.put("status", actionBar.getOrDefault("status", "normal"));
        menu.put("tone", actionBar.getOrDefault("tone", "neutral"));
        menu.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci");
        menu.put("groupCount", groups.size());
        menu.put("itemCount", itemCount);
        menu.put("groups", groups);
        menu.put("actionBar", actionBar);
        return menu;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(String resultFile, Map<String, Object> result) {
        Object menuResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenuCi");
        Map<String, Object> menu = menuResult instanceof Map ? asMap((Map<?, ?>) menuResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu(resultFile, result);
        List<Map<String, Object>> controls = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checksumToolbarControl("openMenu", "Open menu", "button", "primary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu:ci"),
                checksumToolbarControl("search", "Search checksums", "search", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci"),
                checksumToolbarControl("filterStatus", "Filter status", "filter", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummary:ci"),
                checksumToolbarControl("viewCard", "View card", "button", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci"),
                checksumToolbarControl("exportJsonl", "Export JSONL", "button", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci")});
        Map<String, Object> toolbar = new LinkedHashMap<>();
        toolbar.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar");
        toolbar.put("uploadFilesChecksumsToolbarVersion", "checklist-dashboard-gate-upload-files-checksums-toolbar/v1");
        toolbar.put("label", "Upload checksum toolbar");
        toolbar.put("status", menu.getOrDefault("status", "normal"));
        toolbar.put("tone", menu.getOrDefault("tone", "neutral"));
        toolbar.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu:ci");
        toolbar.put("controlCount", controls.size());
        toolbar.put("controls", controls);
        toolbar.put("menuGroupCount", menu.getOrDefault("groupCount", 0));
        toolbar.put("menuItemCount", menu.getOrDefault("itemCount", 0));
        toolbar.put("menu", menu);
        return toolbar;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(String resultFile, Map<String, Object> result) {
        Object toolbarResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbarCi");
        Map<String, Object> toolbar = toolbarResult instanceof Map ? asMap((Map<?, ?>) toolbarResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(resultFile, result);
        List<Map<String, Object>> commands = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checksumCommand("open-card", "Open upload checksum card", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci", "card", "summary", "dashboard"),
                checksumCommand("open-menu", "Open upload checksum menu", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMenu:ci", "menu", "navigation", "groups"),
                checksumCommand("open-toolbar", "Open upload checksum toolbar", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar:ci", "toolbar", "controls", "quick"),
                checksumCommand("show-gauge", "Show checksum gauge", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsGauge:ci", "gauge", "percent", "status"),
                checksumCommand("show-trend", "Show checksum trend", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTrend:ci", "trend", "delta", "baseline"),
                checksumCommand("show-raw", "Show raw checksums", "integrity", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci", "raw", "sha256", "integrity"),
                checksumCommand("copy-jsonl", "Copy checksum JSONL", "logs", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine:ci", "jsonl", "copy", "logs"),
                checksumCommand("copy-markdown", "Copy checksum markdown", "logs", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown:ci", "markdown", "table", "copy"),
                checksumCommand("open-bundle", "Open upload bundle", "upload", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci", "bundle", "upload", "handoff")});
        Map<String, Object> palette = new LinkedHashMap<>();
        palette.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette");
        palette.put("uploadFilesChecksumsCommandPaletteVersion", "checklist-dashboard-gate-upload-files-checksums-command-palette/v1");
        palette.put("label", "Upload checksum command palette");
        palette.put("status", toolbar.getOrDefault("status", "normal"));
        palette.put("tone", toolbar.getOrDefault("tone", "neutral"));
        palette.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar:ci");
        palette.put("commandCount", commands.size());
        palette.put("commands", commands);
        palette.put("toolbarControlCount", toolbar.getOrDefault("controlCount", 0));
        palette.put("menuGroupCount", toolbar.getOrDefault("menuGroupCount", 0));
        palette.put("menuItemCount", toolbar.getOrDefault("menuItemCount", 0));
        palette.put("toolbar", toolbar);
        return palette;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(String resultFile, Map<String, Object> result) {
        Object paletteResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPaletteCi");
        Map<String, Object> palette = paletteResult instanceof Map ? asMap((Map<?, ?>) paletteResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette(resultFile, result);
        List<Map<String, Object>> entries = new ArrayList<>();
        Object commands = palette.get("commands");
        if (commands instanceof List) {
            for (Object commandObject : (List<?>) commands) {
                if (commandObject instanceof Map) {
                    Map<String, Object> command = asMap((Map<?, ?>) commandObject);
                    entries.add(checksumSearchEntry(command));
                }
            }
        }
        Map<String, Object> index = new LinkedHashMap<>();
        index.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex");
        index.put("uploadFilesChecksumsSearchIndexVersion", "checklist-dashboard-gate-upload-files-checksums-search-index/v1");
        index.put("label", "Upload checksum search index");
        index.put("status", palette.getOrDefault("status", "normal"));
        index.put("tone", palette.getOrDefault("tone", "neutral"));
        index.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCommandPalette:ci");
        index.put("entryCount", entries.size());
        index.put("entries", entries);
        index.put("commandPalette", palette);
        return index;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(String resultFile, Map<String, Object> result) {
        Object indexResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndexCi");
        Map<String, Object> index = indexResult instanceof Map ? asMap((Map<?, ?>) indexResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(resultFile, result);
        String query = "checksum";
        List<Map<String, Object>> matches = new ArrayList<>();
        Object entries = index.get("entries");
        if (entries instanceof List) {
            for (Object entryObject : (List<?>) entries) {
                if (entryObject instanceof Map) {
                    Map<String, Object> entry = asMap((Map<?, ?>) entryObject);
                    String text = String.valueOf(entry.getOrDefault("text", "")).toLowerCase(Locale.ROOT);
                    if (text.contains(query)) {
                        matches.add(entry);
                    }
                }
            }
        }
        Map<String, Object> results = new LinkedHashMap<>();
        results.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults");
        results.put("uploadFilesChecksumsSearchResultsVersion", "checklist-dashboard-gate-upload-files-checksums-search-results/v1");
        results.put("label", "Upload checksum search results");
        results.put("status", index.getOrDefault("status", "normal"));
        results.put("tone", index.getOrDefault("tone", "neutral"));
        results.put("query", query);
        results.put("entryCount", index.getOrDefault("entryCount", 0));
        results.put("matchCount", matches.size());
        results.put("matches", matches);
        results.put("searchIndexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex:ci");
        results.put("searchIndex", index);
        return results;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(String resultFile, Map<String, Object> result) {
        Object resultsObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResultsCi");
        Map<String, Object> searchResults = resultsObject instanceof Map ? asMap((Map<?, ?>) resultsObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(resultFile, result);
        Map<String, Object> searchIndex = asMap((Map<?, ?>) searchResults.getOrDefault("searchIndex", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex(resultFile, result)));
        Map<String, Integer> groupCounts = checksumGroupCounts(searchIndex.get("entries"));
        Map<String, Integer> activeGroupCounts = checksumGroupCounts(searchResults.get("matches"));
        Map<String, Object> filterState = new LinkedHashMap<>();
        filterState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState");
        filterState.put("uploadFilesChecksumsFilterStateVersion", "checklist-dashboard-gate-upload-files-checksums-filter-state/v1");
        filterState.put("label", "Upload checksum filter state");
        filterState.put("status", searchResults.getOrDefault("status", "normal"));
        filterState.put("tone", searchResults.getOrDefault("tone", "neutral"));
        filterState.put("query", searchResults.getOrDefault("query", ""));
        filterState.put("selectedGroup", "all");
        filterState.put("selectedStatus", searchResults.getOrDefault("status", "normal"));
        filterState.put("availableGroups", groupCounts.keySet());
        filterState.put("availableStatuses", List.of(searchResults.getOrDefault("status", "normal")));
        filterState.put("groupCounts", groupCounts);
        filterState.put("activeGroupCounts", activeGroupCounts);
        filterState.put("entryCount", searchResults.getOrDefault("entryCount", 0));
        filterState.put("matchCount", searchResults.getOrDefault("matchCount", 0));
        filterState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci");
        filterState.put("searchIndexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex:ci");
        filterState.put("searchResults", searchResults);
        return filterState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(String resultFile, Map<String, Object> result) {
        Object filterObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterStateCi");
        Map<String, Object> filterState = filterObject instanceof Map ? asMap((Map<?, ?>) filterObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState(resultFile, result);
        Object toolbarObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbarCi");
        Map<String, Object> toolbar = toolbarObject instanceof Map ? asMap((Map<?, ?>) toolbarObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar(resultFile, result);
        Map<String, Object> searchResults = asMap((Map<?, ?>) filterState.getOrDefault("searchResults", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults(resultFile, result)));
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("toolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToolbar:ci");
        selectors.put("filterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci");
        selectors.put("searchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci");
        selectors.put("searchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchIndex:ci");
        selectors.put("summaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSummaryCard:ci");
        Map<String, Object> viewState = new LinkedHashMap<>();
        viewState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState");
        viewState.put("uploadFilesChecksumsViewStateVersion", "checklist-dashboard-gate-upload-files-checksums-view-state/v1");
        viewState.put("label", "Upload checksum view state");
        viewState.put("status", filterState.getOrDefault("status", "normal"));
        viewState.put("tone", filterState.getOrDefault("tone", "neutral"));
        viewState.put("activeView", "search");
        viewState.put("activePanel", "results");
        viewState.put("query", filterState.getOrDefault("query", ""));
        viewState.put("selectedGroup", filterState.getOrDefault("selectedGroup", "all"));
        viewState.put("selectedStatus", filterState.getOrDefault("selectedStatus", "normal"));
        viewState.put("entryCount", filterState.getOrDefault("entryCount", 0));
        viewState.put("matchCount", filterState.getOrDefault("matchCount", 0));
        viewState.put("toolbarControlCount", toolbar.getOrDefault("controlCount", 0));
        viewState.put("selectors", selectors);
        viewState.put("toolbar", toolbar);
        viewState.put("filterState", filterState);
        viewState.put("searchResults", searchResults);
        return viewState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(resultFile, result);
        Map<String, Object> emptyState = new LinkedHashMap<>();
        emptyState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState");
        emptyState.put("uploadFilesChecksumsEmptyStateVersion", "checklist-dashboard-gate-upload-files-checksums-empty-state/v1");
        emptyState.put("label", "Upload checksum empty state");
        emptyState.put("status", viewState.getOrDefault("status", "normal"));
        emptyState.put("tone", "muted");
        emptyState.put("emptyReason", "noMatches");
        emptyState.put("query", "no-match-checksum-query");
        emptyState.put("title", "No checksum commands matched");
        emptyState.put("body", "Try clearing filters or searching for checksum, jsonl, markdown, gauge, trend, or bundle.");
        emptyState.put("entryCount", viewState.getOrDefault("entryCount", 0));
        emptyState.put("matchCount", 0);
        emptyState.put("suggestions", List.of("checksum", "jsonl", "markdown", "gauge", "trend", "bundle"));
        emptyState.put("actions", List.of(
                checksumEmptyAction("resetFilters", "Reset filters", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci"),
                checksumEmptyAction("openAllResults", "Open results", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci"),
                checksumEmptyAction("openViewState", "Open view state", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci")));
        emptyState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci");
        emptyState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci");
        emptyState.put("filterStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci");
        emptyState.put("viewState", viewState);
        return emptyState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(resultFile, result);
        Map<String, Object> loadingState = new LinkedHashMap<>();
        loadingState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState");
        loadingState.put("uploadFilesChecksumsLoadingStateVersion", "checklist-dashboard-gate-upload-files-checksums-loading-state/v1");
        loadingState.put("label", "Upload checksum loading state");
        loadingState.put("status", viewState.getOrDefault("status", "normal"));
        loadingState.put("tone", "working");
        loadingState.put("isLoading", true);
        loadingState.put("loadingReason", "refreshingSearch");
        loadingState.put("title", "Refreshing checksum commands");
        loadingState.put("body", "Loading checksum search results and dashboard controls.");
        loadingState.put("query", viewState.getOrDefault("query", ""));
        loadingState.put("entryCount", viewState.getOrDefault("entryCount", 0));
        loadingState.put("expectedMatchCount", viewState.getOrDefault("matchCount", 0));
        loadingState.put("skeletonRowCount", 3);
        loadingState.put("skeletonRows", List.of(
                checksumLoadingSkeletonRow("toolbar", "Toolbar controls"),
                checksumLoadingSkeletonRow("filters", "Filter chips"),
                checksumLoadingSkeletonRow("results", "Search result rows")));
        loadingState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci");
        loadingState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci");
        loadingState.put("filterStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsFilterState:ci");
        loadingState.put("emptyStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState:ci");
        loadingState.put("viewState", viewState);
        return loadingState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(resultFile, result);
        Map<String, Object> errorState = new LinkedHashMap<>();
        errorState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState");
        errorState.put("uploadFilesChecksumsErrorStateVersion", "checklist-dashboard-gate-upload-files-checksums-error-state/v1");
        errorState.put("label", "Upload checksum error state");
        errorState.put("status", "error");
        errorState.put("tone", "critical");
        errorState.put("severity", "recoverable");
        errorState.put("errorCode", "checksumSearchRefreshFailed");
        errorState.put("title", "Checksum search could not refresh");
        errorState.put("body", "Retry the checksum search or open the last known view state.");
        errorState.put("query", viewState.getOrDefault("query", ""));
        errorState.put("entryCount", viewState.getOrDefault("entryCount", 0));
        errorState.put("lastKnownMatchCount", viewState.getOrDefault("matchCount", 0));
        errorState.put("actions", List.of(
                checksumStateAction("retry", "Retry refresh", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci"),
                checksumStateAction("openViewState", "Open last known view", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci"),
                checksumStateAction("openSearchResults", "Open search results", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci")));
        errorState.put("loadingStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci");
        errorState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci");
        errorState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci");
        errorState.put("emptyStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEmptyState:ci");
        errorState.put("viewState", viewState);
        return errorState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState(resultFile, result);
        Map<String, Object> toastState = new LinkedHashMap<>();
        toastState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState");
        toastState.put("uploadFilesChecksumsToastStateVersion", "checklist-dashboard-gate-upload-files-checksums-toast-state/v1");
        toastState.put("label", "Upload checksum toast state");
        toastState.put("status", viewState.getOrDefault("status", "normal"));
        toastState.put("tone", "informational");
        toastState.put("placement", "bottom-right");
        toastState.put("defaultToastId", "checksums-refreshed");
        toastState.put("toastCount", 3);
        toastState.put("toasts", List.of(
                checksumToast("checksums-refreshed", "success", "Checksums refreshed", "8 checksum matches are ready.", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci"),
                checksumToast("checksums-loading", "info", "Refreshing checksums", "Search results are being refreshed.", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci"),
                checksumToast("checksums-retry", "error", "Checksum refresh failed", "Retry refresh or open the last known view.", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci")));
        toastState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci");
        toastState.put("loadingStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci");
        toastState.put("errorStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci");
        toastState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsSearchResults:ci");
        toastState.put("viewState", viewState);
        return toastState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(String resultFile, Map<String, Object> result) {
        Object toastObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastStateCi");
        Map<String, Object> toastState = toastObject instanceof Map ? asMap((Map<?, ?>) toastObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState(resultFile, result);
        Object toastsObject = toastState.getOrDefault("toasts", List.of());
        List<?> toasts = toastsObject instanceof List ? (List<?>) toastsObject : List.of();
        Map<String, Object> notificationCenter = new LinkedHashMap<>();
        notificationCenter.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter");
        notificationCenter.put("uploadFilesChecksumsNotificationCenterVersion", "checklist-dashboard-gate-upload-files-checksums-notification-center/v1");
        notificationCenter.put("label", "Upload checksum notification center");
        notificationCenter.put("status", toastState.getOrDefault("status", "normal"));
        notificationCenter.put("tone", "informational");
        notificationCenter.put("activeNotificationId", toastState.getOrDefault("defaultToastId", ""));
        notificationCenter.put("notificationCount", toasts.size());
        notificationCenter.put("unreadCount", 2);
        notificationCenter.put("errorCount", 1);
        notificationCenter.put("groups", List.of(
                checksumNotificationGroup("recent", "Recent", 2),
                checksumNotificationGroup("attention", "Needs attention", 1)));
        notificationCenter.put("notifications", toasts);
        notificationCenter.put("toastStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci");
        notificationCenter.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsViewState:ci");
        notificationCenter.put("loadingStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsLoadingState:ci");
        notificationCenter.put("errorStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci");
        notificationCenter.put("toastState", toastState);
        return notificationCenter;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(String resultFile, Map<String, Object> result) {
        Object centerObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenterCi");
        Map<String, Object> notificationCenter = centerObject instanceof Map ? asMap((Map<?, ?>) centerObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(resultFile, result);
        int unreadCount = numberAsInt(notificationCenter.getOrDefault("unreadCount", 0));
        int errorCount = numberAsInt(notificationCenter.getOrDefault("errorCount", 0));
        Map<String, Object> badge = new LinkedHashMap<>();
        badge.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge");
        badge.put("uploadFilesChecksumsNotificationBadgeVersion", "checklist-dashboard-gate-upload-files-checksums-notification-badge/v1");
        badge.put("label", unreadCount + " checksum notifications");
        badge.put("status", notificationCenter.getOrDefault("status", "normal"));
        badge.put("tone", errorCount > 0 ? "critical" : "informational");
        badge.put("visible", unreadCount > 0);
        badge.put("unreadCount", unreadCount);
        badge.put("errorCount", errorCount);
        badge.put("notificationCount", notificationCenter.getOrDefault("notificationCount", 0));
        badge.put("activeNotificationId", notificationCenter.getOrDefault("activeNotificationId", ""));
        badge.put("notificationCenterSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci");
        badge.put("toastStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci");
        badge.put("errorStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci");
        badge.put("notificationCenter", notificationCenter);
        return badge;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(String resultFile, Map<String, Object> result) {
        Object badgeObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadgeCi");
        Map<String, Object> badge = badgeObject instanceof Map ? asMap((Map<?, ?>) badgeObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(resultFile, result);
        Object centerObject = badge.get("notificationCenter");
        Map<String, Object> notificationCenter = centerObject instanceof Map ? asMap((Map<?, ?>) centerObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter(resultFile, result);
        Object notificationsObject = notificationCenter.getOrDefault("notifications", List.of());
        List<?> notifications = notificationsObject instanceof List ? (List<?>) notificationsObject : List.of();
        Object groupsObject = notificationCenter.getOrDefault("groups", List.of());
        List<?> groups = groupsObject instanceof List ? (List<?>) groupsObject : List.of();
        Map<String, Object> panel = new LinkedHashMap<>();
        panel.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel");
        panel.put("uploadFilesChecksumsNotificationPanelVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel/v1");
        panel.put("label", "Upload checksum notifications");
        panel.put("status", badge.getOrDefault("status", "normal"));
        panel.put("tone", badge.getOrDefault("tone", "informational"));
        panel.put("expanded", true);
        panel.put("placement", "header-popover");
        panel.put("topNotificationCount", Math.min(3, notifications.size()));
        panel.put("groupCount", groups.size());
        panel.put("topNotifications", notifications.size() > 3 ? new ArrayList<>(notifications.subList(0, 3)) : notifications);
        panel.put("groups", groups);
        panel.put("badge", badge);
        panel.put("notificationCenterSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci");
        panel.put("notificationBadgeSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci");
        panel.put("toastStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci");
        panel.put("errorStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci");
        panel.put("notificationCenter", notificationCenter);
        return panel;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLine(String resultFile, Map<String, Object> result) {
        Object panelObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        Map<String, Object> panel = panelObject instanceof Map ? asMap((Map<?, ?>) panelObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile, result);
        return "checklistDashboardGateUploadFilesChecksumsNotificationPanel"
                + " status=" + panel.getOrDefault("status", "normal")
                + " tone=" + panel.getOrDefault("tone", "informational")
                + " expanded=" + panel.getOrDefault("expanded", false)
                + " placement=" + panel.getOrDefault("placement", "header-popover")
                + " topNotificationCount=" + panel.getOrDefault("topNotificationCount", 0)
                + " groupCount=" + panel.getOrDefault("groupCount", 0)
                + " notificationCenterSelector=" + panel.getOrDefault("notificationCenterSelector", "")
                + " notificationBadgeSelector=" + panel.getOrDefault("notificationBadgeSelector", "");
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine(String resultFile, Map<String, Object> result) {
        Object panelObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        Map<String, Object> panel = panelObject instanceof Map ? asMap((Map<?, ?>) panelObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile, result);
        Map<String, Object> jsonLine = new LinkedHashMap<>();
        jsonLine.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine");
        jsonLine.put("uploadFilesChecksumsNotificationPanelJsonLineVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-json-line/v1");
        jsonLine.put("status", panel.getOrDefault("status", "normal"));
        jsonLine.put("tone", panel.getOrDefault("tone", "informational"));
        jsonLine.put("expanded", panel.getOrDefault("expanded", false));
        jsonLine.put("placement", panel.getOrDefault("placement", "header-popover"));
        jsonLine.put("topNotificationCount", panel.getOrDefault("topNotificationCount", 0));
        jsonLine.put("groupCount", panel.getOrDefault("groupCount", 0));
        jsonLine.put("panelSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci");
        jsonLine.put("notificationCenterSelector", panel.getOrDefault("notificationCenterSelector", ""));
        jsonLine.put("notificationBadgeSelector", panel.getOrDefault("notificationBadgeSelector", ""));
        try {
            return JSON_LINE.writeValueAsString(jsonLine);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to render gate upload checksum notification panel JSON line.", exception);
        }
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine(String resultFile, Map<String, Object> result) {
        Object panelObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        Map<String, Object> panel = panelObject instanceof Map ? asMap((Map<?, ?>) panelObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile, result);
        List<String> headers = List.of("summary", "csvLineVersion", "status", "tone", "expanded", "placement", "topNotificationCount", "groupCount", "panelSelector", "notificationCenterSelector", "notificationBadgeSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCsvLine",
                "checklist-dashboard-gate-upload-files-checksums-notification-panel-csv-line/v1",
                String.valueOf(panel.getOrDefault("status", "normal")),
                String.valueOf(panel.getOrDefault("tone", "informational")),
                String.valueOf(panel.getOrDefault("expanded", false)),
                String.valueOf(panel.getOrDefault("placement", "header-popover")),
                String.valueOf(panel.getOrDefault("topNotificationCount", 0)),
                String.valueOf(panel.getOrDefault("groupCount", 0)),
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci",
                String.valueOf(panel.getOrDefault("notificationCenterSelector", "")),
                String.valueOf(panel.getOrDefault("notificationBadgeSelector", "")));
        return String.join(",", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList()) + "\n"
                + String.join(",", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine(String resultFile, Map<String, Object> result) {
        Object panelObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        Map<String, Object> panel = panelObject instanceof Map ? asMap((Map<?, ?>) panelObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile, result);
        List<String> headers = List.of("summary", "tsvLineVersion", "status", "tone", "expanded", "placement", "topNotificationCount", "groupCount", "panelSelector", "notificationCenterSelector", "notificationBadgeSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelTsvLine",
                "checklist-dashboard-gate-upload-files-checksums-notification-panel-tsv-line/v1",
                String.valueOf(panel.getOrDefault("status", "normal")),
                String.valueOf(panel.getOrDefault("tone", "informational")),
                String.valueOf(panel.getOrDefault("expanded", false)),
                String.valueOf(panel.getOrDefault("placement", "header-popover")),
                String.valueOf(panel.getOrDefault("topNotificationCount", 0)),
                String.valueOf(panel.getOrDefault("groupCount", 0)),
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci",
                String.valueOf(panel.getOrDefault("notificationCenterSelector", "")),
                String.valueOf(panel.getOrDefault("notificationBadgeSelector", "")));
        return String.join("\t", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList()) + "\n"
                + String.join("\t", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown(String resultFile, Map<String, Object> result) {
        Object panelObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        Map<String, Object> panel = panelObject instanceof Map ? asMap((Map<?, ?>) panelObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile, result);
        Object notificationsObject = panel.getOrDefault("topNotifications", List.of());
        List<?> notifications = notificationsObject instanceof List ? (List<?>) notificationsObject : List.of();
        List<String> lines = new ArrayList<>();
        lines.add("**Checklist dashboard gate upload checksum notifications**");
        lines.add("");
        lines.add("Status: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("status", "normal"))) + "`"
                + " | Tone: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("tone", "informational"))) + "`"
                + " | Expanded: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("expanded", false))) + "`"
                + " | Placement: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("placement", "header-popover"))) + "`");
        lines.add("Top notifications: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("topNotificationCount", 0))) + "`"
                + " | Groups: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("groupCount", 0))) + "`");
        lines.add("");
        lines.add("| ID | Tone | Title | Selector |");
        lines.add("| --- | --- | --- | --- |");
        for (Object notificationObject : notifications) {
            if (!(notificationObject instanceof Map)) {
                continue;
            }
            Map<String, Object> notification = asMap((Map<?, ?>) notificationObject);
            lines.add("| " + escapeMarkdownTableCell(String.valueOf(notification.getOrDefault("id", "")))
                    + " | " + escapeMarkdownTableCell(String.valueOf(notification.getOrDefault("tone", "")))
                    + " | " + escapeMarkdownTableCell(String.valueOf(notification.getOrDefault("title", "")))
                    + " | `" + escapeMarkdownTableCode(String.valueOf(notification.getOrDefault("selector", ""))) + "` |");
        }
        lines.add("");
        lines.add("Panel selector: `resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci`");
        lines.add("Notification center selector: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("notificationCenterSelector", ""))) + "`");
        lines.add("Notification badge selector: `" + escapeMarkdownTableCode(String.valueOf(panel.getOrDefault("notificationBadgeSelector", ""))) + "`");
        return String.join("\n", lines);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv(String resultFile, Map<String, Object> result) {
        Object panelObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        Map<String, Object> panel = panelObject instanceof Map ? asMap((Map<?, ?>) panelObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile, result);
        Object notificationsObject = panel.getOrDefault("topNotifications", List.of());
        List<?> notifications = notificationsObject instanceof List ? (List<?>) notificationsObject : List.of();
        List<String> notificationIds = new ArrayList<>();
        for (Object notificationObject : notifications) {
            if (!(notificationObject instanceof Map)) {
                continue;
            }
            Map<String, Object> notification = asMap((Map<?, ?>) notificationObject);
            String id = String.valueOf(notification.getOrDefault("id", ""));
            if (!id.isBlank()) {
                notificationIds.add(id);
            }
        }
        List<String> lines = new ArrayList<>();
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_ENV_VERSION=" + shellQuote("checklist-dashboard-gate-upload-files-checksums-notification-panel-env/v1"));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_STATUS=" + shellQuote(String.valueOf(panel.getOrDefault("status", "normal"))));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_TONE=" + shellQuote(String.valueOf(panel.getOrDefault("tone", "informational"))));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_EXPANDED=" + shellQuote(String.valueOf(panel.getOrDefault("expanded", false))));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_PLACEMENT=" + shellQuote(String.valueOf(panel.getOrDefault("placement", "header-popover"))));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_TOP_COUNT=" + panel.getOrDefault("topNotificationCount", notificationIds.size()));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_GROUP_COUNT=" + panel.getOrDefault("groupCount", 0));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_TOP_IDS=" + shellQuote(String.join(",", notificationIds)));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_PANEL_SELECTOR=" + shellQuote("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci"));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_CENTER_SELECTOR=" + shellQuote(String.valueOf(panel.getOrDefault("notificationCenterSelector", ""))));
        lines.add("GATE_UPLOAD_CHECKSUM_NOTIFICATION_BADGE_SELECTOR=" + shellQuote(String.valueOf(panel.getOrDefault("notificationBadgeSelector", ""))));
        return String.join("\n", lines);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(String resultFile, Map<String, Object> result) {
        Object panelObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCi");
        Map<String, Object> panel = panelObject instanceof Map ? asMap((Map<?, ?>) panelObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel(resultFile, result);
        Object badgeObject = panel.get("badge");
        Map<String, Object> badge = badgeObject instanceof Map ? asMap((Map<?, ?>) badgeObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge(resultFile, result);
        Object notificationsObject = panel.getOrDefault("topNotifications", List.of());
        List<?> notifications = notificationsObject instanceof List ? (List<?>) notificationsObject : List.of();
        String subtitle = panel.getOrDefault("topNotificationCount", 0) + " top notifications / " + panel.getOrDefault("groupCount", 0) + " groups";
        Map<String, Object> card = new LinkedHashMap<>();
        card.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard");
        card.put("uploadFilesChecksumsNotificationPanelSummaryCardVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-summary-card/v1");
        card.put("title", "Upload checksum notifications");
        card.put("subtitle", subtitle);
        card.put("status", panel.getOrDefault("status", "normal"));
        card.put("tone", panel.getOrDefault("tone", "informational"));
        card.put("badgeLabel", badge.getOrDefault("label", ""));
        card.put("expanded", panel.getOrDefault("expanded", false));
        card.put("placement", panel.getOrDefault("placement", "header-popover"));
        card.put("topNotificationCount", panel.getOrDefault("topNotificationCount", 0));
        card.put("groupCount", panel.getOrDefault("groupCount", 0));
        card.put("activeNotificationId", badge.getOrDefault("activeNotificationId", ""));
        card.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci");
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("panel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci");
        selectors.put("badge", panel.getOrDefault("notificationBadgeSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci"));
        selectors.put("center", panel.getOrDefault("notificationCenterSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci"));
        selectors.put("env", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci");
        selectors.put("markdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci");
        card.put("selectors", selectors);
        card.put("topNotifications", notifications);
        card.put("panel", panel);
        return card;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(String resultFile, Map<String, Object> result) {
        Object cardObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCardCi");
        Map<String, Object> card = cardObject instanceof Map ? asMap((Map<?, ?>) cardObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard(resultFile, result);
        String tone = "critical".equals(String.valueOf(card.getOrDefault("tone", "informational"))) ? "attention" : "neutral";
        Object selectorsObject = card.get("selectors");
        Map<String, Object> selectors = selectorsObject instanceof Map ? asMap((Map<?, ?>) selectorsObject) : Map.of();
        List<Map<String, Object>> actions = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checklistAction("openPanel", "Open panel", "primary", tone, "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci"),
                checklistAction("openCenter", "Open center", "secondary", "neutral", String.valueOf(selectors.getOrDefault("center", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci"))),
                checklistAction("copyMarkdown", "Copy markdown", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci"),
                checklistAction("copyEnv", "Copy env", "secondary", "neutral", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci"),
                checklistAction("openBadge", "Open badge", "secondary", "neutral", String.valueOf(selectors.getOrDefault("badge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci")))});
        Map<String, Object> actionBar = new LinkedHashMap<>();
        actionBar.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar");
        actionBar.put("uploadFilesChecksumsNotificationPanelActionBarVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-action-bar/v1");
        actionBar.put("label", "Upload checksum notification actions");
        actionBar.put("status", card.getOrDefault("status", "normal"));
        actionBar.put("tone", tone);
        actionBar.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci");
        actionBar.put("actionCount", actions.size());
        actionBar.put("actions", actions);
        actionBar.put("summaryCard", card);
        return actionBar;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(String resultFile, Map<String, Object> result) {
        Object actionBarObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBarCi");
        Map<String, Object> actionBar = actionBarObject instanceof Map ? asMap((Map<?, ?>) actionBarObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar(resultFile, result);
        List<Map<String, Object>> groups = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checksumMenuGroup("overview", "Overview", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                        checksumMenuItem("summaryCard", "Summary card", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard:ci"),
                        checksumMenuItem("panel", "Panel", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci"),
                        checksumMenuItem("badge", "Badge", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci")})),
                checksumMenuGroup("actions", "Actions", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                        checksumMenuItem("actionBar", "Action bar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar:ci"),
                        checksumMenuItem("center", "Notification center", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci"),
                        checksumMenuItem("toastState", "Toast state", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsToastState:ci")})),
                checksumMenuGroup("exports", "Exports", List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                        checksumMenuItem("markdown", "Markdown", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci"),
                        checksumMenuItem("env", "Environment", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci"),
                        checksumMenuItem("jsonLine", "JSONL", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine:ci")}))});
        Map<String, Object> menu = new LinkedHashMap<>();
        menu.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu");
        menu.put("uploadFilesChecksumsNotificationPanelMenuVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-menu/v1");
        menu.put("label", "Upload checksum notification menu");
        menu.put("status", actionBar.getOrDefault("status", "normal"));
        menu.put("tone", actionBar.getOrDefault("tone", "neutral"));
        menu.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelActionBar:ci");
        menu.put("groupCount", groups.size());
        menu.put("itemCount", groups.stream().mapToInt(group -> {
            Object items = group.get("items");
            return items instanceof List ? ((List<?>) items).size() : 0;
        }).sum());
        menu.put("groups", groups);
        menu.put("actionBar", actionBar);
        return menu;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(String resultFile, Map<String, Object> result) {
        Object menuObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenuCi");
        Map<String, Object> menu = menuObject instanceof Map ? asMap((Map<?, ?>) menuObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu(resultFile, result);
        List<Map<String, Object>> controls = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checksumToolbarControl("openMenu", "Open menu", "button", "primary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu:ci"),
                checksumToolbarControl("searchNotifications", "Search notifications", "search", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci"),
                checksumToolbarControl("filterAttention", "Filter attention", "filter", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci"),
                checksumToolbarControl("copyMarkdown", "Copy markdown", "button", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci"),
                checksumToolbarControl("copyEnv", "Copy env", "button", "secondary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci")});
        Map<String, Object> toolbar = new LinkedHashMap<>();
        toolbar.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar");
        toolbar.put("uploadFilesChecksumsNotificationPanelToolbarVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-toolbar/v1");
        toolbar.put("label", "Upload checksum notification toolbar");
        toolbar.put("status", menu.getOrDefault("status", "normal"));
        toolbar.put("tone", menu.getOrDefault("tone", "neutral"));
        toolbar.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu:ci");
        toolbar.put("controlCount", controls.size());
        toolbar.put("controls", controls);
        toolbar.put("menuGroupCount", menu.getOrDefault("groupCount", 0));
        toolbar.put("menuItemCount", menu.getOrDefault("itemCount", 0));
        toolbar.put("menu", menu);
        return toolbar;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(String resultFile, Map<String, Object> result) {
        Object toolbarObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbarCi");
        Map<String, Object> toolbar = toolbarObject instanceof Map ? asMap((Map<?, ?>) toolbarObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(resultFile, result);
        List<Map<String, Object>> commands = List.<Map<String, Object>>of((Map<String, Object>[]) new Map[]{
                checksumCommand("open-panel", "Open notification panel", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanel:ci", "panel", "notifications", "open"),
                checksumCommand("open-menu", "Open notification menu", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMenu:ci", "menu", "navigation", "groups"),
                checksumCommand("open-toolbar", "Open notification toolbar", "overview", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar:ci", "toolbar", "controls", "quick"),
                checksumCommand("open-center", "Open notification center", "actions", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationCenter:ci", "center", "history", "notifications"),
                checksumCommand("open-badge", "Open notification badge", "actions", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationBadge:ci", "badge", "count", "attention"),
                checksumCommand("copy-markdown", "Copy notification markdown", "exports", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelMarkdown:ci", "markdown", "copy", "export"),
                checksumCommand("copy-env", "Copy notification env", "exports", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEnv:ci", "env", "shell", "export"),
                checksumCommand("copy-jsonl", "Copy notification JSONL", "exports", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelJsonLine:ci", "jsonl", "logs", "export"),
                checksumCommand("show-attention", "Show attention notifications", "actions", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsErrorState:ci", "attention", "error", "filter")});
        Map<String, Object> palette = new LinkedHashMap<>();
        palette.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette");
        palette.put("uploadFilesChecksumsNotificationPanelCommandPaletteVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-command-palette/v1");
        palette.put("label", "Upload checksum notification command palette");
        palette.put("status", toolbar.getOrDefault("status", "normal"));
        palette.put("tone", toolbar.getOrDefault("tone", "neutral"));
        palette.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar:ci");
        palette.put("commandCount", commands.size());
        palette.put("commands", commands);
        palette.put("toolbarControlCount", toolbar.getOrDefault("controlCount", 0));
        palette.put("menuGroupCount", toolbar.getOrDefault("menuGroupCount", 0));
        palette.put("menuItemCount", toolbar.getOrDefault("menuItemCount", 0));
        palette.put("toolbar", toolbar);
        return palette;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(String resultFile, Map<String, Object> result) {
        Object paletteResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPaletteCi");
        Map<String, Object> palette = paletteResult instanceof Map ? asMap((Map<?, ?>) paletteResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette(resultFile, result);
        List<Map<String, Object>> entries = new ArrayList<>();
        Object commands = palette.get("commands");
        if (commands instanceof List) {
            for (Object commandObject : (List<?>) commands) {
                if (commandObject instanceof Map) {
                    Map<String, Object> command = asMap((Map<?, ?>) commandObject);
                    entries.add(checksumSearchEntry(command));
                }
            }
        }
        Map<String, Object> index = new LinkedHashMap<>();
        index.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex");
        index.put("uploadFilesChecksumsNotificationPanelSearchIndexVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-search-index/v1");
        index.put("label", "Upload checksum notification search index");
        index.put("status", palette.getOrDefault("status", "normal"));
        index.put("tone", palette.getOrDefault("tone", "neutral"));
        index.put("primarySelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci");
        index.put("entryCount", entries.size());
        index.put("entries", entries);
        index.put("commandPalette", palette);
        return index;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(String resultFile, Map<String, Object> result) {
        Object indexResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndexCi");
        Map<String, Object> index = indexResult instanceof Map ? asMap((Map<?, ?>) indexResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(resultFile, result);
        String query = "notification";
        List<Map<String, Object>> matches = new ArrayList<>();
        Object entries = index.get("entries");
        if (entries instanceof List) {
            for (Object entryObject : (List<?>) entries) {
                if (entryObject instanceof Map) {
                    Map<String, Object> entry = asMap((Map<?, ?>) entryObject);
                    String text = String.valueOf(entry.getOrDefault("text", "")).toLowerCase(Locale.ROOT);
                    if (text.contains(query)) {
                        matches.add(entry);
                    }
                }
            }
        }
        Map<String, Object> results = new LinkedHashMap<>();
        results.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults");
        results.put("uploadFilesChecksumsNotificationPanelSearchResultsVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-search-results/v1");
        results.put("label", "Upload checksum notification search results");
        results.put("status", index.getOrDefault("status", "normal"));
        results.put("tone", index.getOrDefault("tone", "neutral"));
        results.put("query", query);
        results.put("entryCount", index.getOrDefault("entryCount", 0));
        results.put("matchCount", matches.size());
        results.put("matches", matches);
        results.put("searchIndexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci");
        results.put("searchIndex", index);
        return results;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(String resultFile, Map<String, Object> result) {
        Object resultsObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResultsCi");
        Map<String, Object> searchResults = resultsObject instanceof Map ? asMap((Map<?, ?>) resultsObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(resultFile, result);
        Map<String, Object> searchIndex = asMap((Map<?, ?>) searchResults.getOrDefault("searchIndex", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex(resultFile, result)));
        Map<String, Integer> groupCounts = checksumGroupCounts(searchIndex.get("entries"));
        Map<String, Integer> activeGroupCounts = checksumGroupCounts(searchResults.get("matches"));
        Map<String, Object> filterState = new LinkedHashMap<>();
        filterState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState");
        filterState.put("uploadFilesChecksumsNotificationPanelFilterStateVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-filter-state/v1");
        filterState.put("label", "Upload checksum notification filter state");
        filterState.put("status", searchResults.getOrDefault("status", "normal"));
        filterState.put("tone", searchResults.getOrDefault("tone", "neutral"));
        filterState.put("query", searchResults.getOrDefault("query", ""));
        filterState.put("selectedGroup", "all");
        filterState.put("selectedStatus", searchResults.getOrDefault("status", "normal"));
        filterState.put("availableGroups", groupCounts.keySet());
        filterState.put("availableStatuses", List.of(searchResults.getOrDefault("status", "normal")));
        filterState.put("groupCounts", groupCounts);
        filterState.put("activeGroupCounts", activeGroupCounts);
        filterState.put("entryCount", searchResults.getOrDefault("entryCount", 0));
        filterState.put("matchCount", searchResults.getOrDefault("matchCount", 0));
        filterState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci");
        filterState.put("searchIndexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci");
        filterState.put("searchResults", searchResults);
        return filterState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(String resultFile, Map<String, Object> result) {
        Object filterObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterStateCi");
        Map<String, Object> filterState = filterObject instanceof Map ? asMap((Map<?, ?>) filterObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState(resultFile, result);
        Object toolbarObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbarCi");
        Map<String, Object> toolbar = toolbarObject instanceof Map ? asMap((Map<?, ?>) toolbarObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar(resultFile, result);
        Map<String, Object> searchResults = asMap((Map<?, ?>) filterState.getOrDefault("searchResults", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults(resultFile, result)));
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("toolbar", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToolbar:ci");
        selectors.put("filterState", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci");
        selectors.put("searchResults", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci");
        selectors.put("searchIndex", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci");
        selectors.put("summaryCard", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSummaryCard:ci");
        Map<String, Object> viewState = new LinkedHashMap<>();
        viewState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState");
        viewState.put("uploadFilesChecksumsNotificationPanelViewStateVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-view-state/v1");
        viewState.put("label", "Upload checksum notification view state");
        viewState.put("status", filterState.getOrDefault("status", "normal"));
        viewState.put("tone", filterState.getOrDefault("tone", "neutral"));
        viewState.put("activeView", "notifications");
        viewState.put("activePanel", "results");
        viewState.put("query", filterState.getOrDefault("query", ""));
        viewState.put("selectedGroup", filterState.getOrDefault("selectedGroup", "all"));
        viewState.put("selectedStatus", filterState.getOrDefault("selectedStatus", "normal"));
        viewState.put("entryCount", filterState.getOrDefault("entryCount", 0));
        viewState.put("matchCount", filterState.getOrDefault("matchCount", 0));
        viewState.put("toolbarControlCount", toolbar.getOrDefault("controlCount", 0));
        viewState.put("selectors", selectors);
        viewState.put("toolbar", toolbar);
        viewState.put("filterState", filterState);
        viewState.put("searchResults", searchResults);
        return viewState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(resultFile, result);
        Map<String, Object> emptyState = new LinkedHashMap<>();
        emptyState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState");
        emptyState.put("uploadFilesChecksumsNotificationPanelEmptyStateVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-empty-state/v1");
        emptyState.put("label", "Upload checksum notification empty state");
        emptyState.put("status", viewState.getOrDefault("status", "normal"));
        emptyState.put("tone", "muted");
        emptyState.put("emptyReason", "noMatchingNotifications");
        emptyState.put("query", "no-match-notification-query");
        emptyState.put("title", "No notification commands matched");
        emptyState.put("body", "Clear notification filters or search for panel, center, badge, markdown, env, or attention.");
        emptyState.put("activeView", viewState.getOrDefault("activeView", "notifications"));
        emptyState.put("activePanel", viewState.getOrDefault("activePanel", "results"));
        emptyState.put("entryCount", viewState.getOrDefault("entryCount", 0));
        emptyState.put("matchCount", 0);
        emptyState.put("suggestions", List.of("panel", "center", "badge", "markdown", "env", "attention"));
        emptyState.put("actions", List.of(
                checksumEmptyAction("resetNotificationFilters", "Reset notification filters", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci"),
                checksumEmptyAction("openNotificationResults", "Open notification results", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci"),
                checksumEmptyAction("openNotificationCommands", "Open command palette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci"),
                checksumEmptyAction("openNotificationViewState", "Open view state", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci")));
        emptyState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci");
        emptyState.put("filterStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci");
        emptyState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci");
        emptyState.put("searchIndexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchIndex:ci");
        emptyState.put("commandPaletteSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci");
        emptyState.put("viewState", viewState);
        return emptyState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(resultFile, result);
        Map<String, Object> loadingState = new LinkedHashMap<>();
        loadingState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState");
        loadingState.put("uploadFilesChecksumsNotificationPanelLoadingStateVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-loading-state/v1");
        loadingState.put("label", "Upload checksum notification loading state");
        loadingState.put("status", viewState.getOrDefault("status", "normal"));
        loadingState.put("tone", "working");
        loadingState.put("isLoading", true);
        loadingState.put("loadingReason", "refreshingNotificationSearch");
        loadingState.put("title", "Refreshing notification commands");
        loadingState.put("body", "Loading notification-panel search results, filters, and toolbar controls.");
        loadingState.put("activeView", viewState.getOrDefault("activeView", "notifications"));
        loadingState.put("activePanel", viewState.getOrDefault("activePanel", "results"));
        loadingState.put("query", viewState.getOrDefault("query", ""));
        loadingState.put("entryCount", viewState.getOrDefault("entryCount", 0));
        loadingState.put("expectedMatchCount", viewState.getOrDefault("matchCount", 0));
        loadingState.put("skeletonRowCount", 4);
        loadingState.put("skeletonRows", List.of(
                checksumLoadingSkeletonRow("toolbar", "Notification toolbar controls"),
                checksumLoadingSkeletonRow("filters", "Notification filter chips"),
                checksumLoadingSkeletonRow("results", "Notification result rows"),
                checksumLoadingSkeletonRow("actions", "Notification command actions")));
        loadingState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci");
        loadingState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci");
        loadingState.put("filterStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelFilterState:ci");
        loadingState.put("emptyStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState:ci");
        loadingState.put("commandPaletteSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci");
        loadingState.put("viewState", viewState);
        return loadingState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(resultFile, result);
        Map<String, Object> errorState = new LinkedHashMap<>();
        errorState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState");
        errorState.put("uploadFilesChecksumsNotificationPanelErrorStateVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-error-state/v1");
        errorState.put("label", "Upload checksum notification error state");
        errorState.put("status", "error");
        errorState.put("tone", "critical");
        errorState.put("severity", "recoverable");
        errorState.put("errorCode", "notificationPanelRefreshFailed");
        errorState.put("title", "Notification panel could not refresh");
        errorState.put("body", "Retry the notification-panel refresh or open the last known notification view state.");
        errorState.put("activeView", viewState.getOrDefault("activeView", "notifications"));
        errorState.put("activePanel", viewState.getOrDefault("activePanel", "results"));
        errorState.put("query", viewState.getOrDefault("query", ""));
        errorState.put("entryCount", viewState.getOrDefault("entryCount", 0));
        errorState.put("lastKnownMatchCount", viewState.getOrDefault("matchCount", 0));
        errorState.put("actions", List.of(
                checksumStateAction("retryNotificationRefresh", "Retry notification refresh", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci"),
                checksumStateAction("openNotificationViewState", "Open last known view", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci"),
                checksumStateAction("openNotificationResults", "Open notification results", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci"),
                checksumStateAction("openNotificationCommands", "Open command palette", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci")));
        errorState.put("loadingStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci");
        errorState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci");
        errorState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci");
        errorState.put("emptyStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState:ci");
        errorState.put("commandPaletteSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelCommandPalette:ci");
        errorState.put("viewState", viewState);
        return errorState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(String resultFile, Map<String, Object> result) {
        Object viewObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewStateCi");
        Map<String, Object> viewState = viewObject instanceof Map ? asMap((Map<?, ?>) viewObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState(resultFile, result);
        Map<String, Object> toastState = new LinkedHashMap<>();
        toastState.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState");
        toastState.put("uploadFilesChecksumsNotificationPanelToastStateVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-toast-state/v1");
        toastState.put("label", "Upload checksum notification toast state");
        toastState.put("status", viewState.getOrDefault("status", "normal"));
        toastState.put("tone", "informational");
        toastState.put("placement", "bottom-right");
        toastState.put("defaultToastId", "notification-panel-refreshed");
        toastState.put("toastCount", 3);
        toastState.put("toasts", List.of(
                checksumToast("notification-panel-refreshed", "success", "Notification panel refreshed", "9 notification commands are ready.", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci"),
                checksumToast("notification-panel-loading", "info", "Refreshing notification panel", "Notification-panel search results are being refreshed.", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci"),
                checksumToast("notification-panel-retry", "error", "Notification panel refresh failed", "Retry refresh or open the last known notification view.", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState:ci")));
        toastState.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci");
        toastState.put("loadingStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci");
        toastState.put("errorStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState:ci");
        toastState.put("searchResultsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelSearchResults:ci");
        toastState.put("emptyStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState:ci");
        toastState.put("viewState", viewState);
        return toastState;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(String resultFile, Map<String, Object> result) {
        Object toastObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastStateCi");
        Map<String, Object> toastState = toastObject instanceof Map ? asMap((Map<?, ?>) toastObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState(resultFile, result);
        Object toastsObject = toastState.getOrDefault("toasts", List.of());
        List<?> toasts = toastsObject instanceof List ? (List<?>) toastsObject : List.of();
        Map<String, Object> notificationCenter = new LinkedHashMap<>();
        notificationCenter.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter");
        notificationCenter.put("uploadFilesChecksumsNotificationPanelNotificationCenterVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-notification-center/v1");
        notificationCenter.put("label", "Upload checksum notification-panel center");
        notificationCenter.put("status", toastState.getOrDefault("status", "normal"));
        notificationCenter.put("tone", "informational");
        notificationCenter.put("activeNotificationId", toastState.getOrDefault("defaultToastId", ""));
        notificationCenter.put("notificationCount", toasts.size());
        notificationCenter.put("unreadCount", 2);
        notificationCenter.put("errorCount", 1);
        notificationCenter.put("groups", List.of(
                checksumNotificationGroup("recent", "Recent panel notifications", 2),
                checksumNotificationGroup("attention", "Needs panel attention", 1)));
        notificationCenter.put("notifications", toasts);
        notificationCenter.put("toastStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState:ci");
        notificationCenter.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci");
        notificationCenter.put("loadingStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelLoadingState:ci");
        notificationCenter.put("errorStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState:ci");
        notificationCenter.put("emptyStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelEmptyState:ci");
        notificationCenter.put("toastState", toastState);
        return notificationCenter;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(String resultFile, Map<String, Object> result) {
        Object centerObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenterCi");
        Map<String, Object> notificationCenter = centerObject instanceof Map ? asMap((Map<?, ?>) centerObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter(resultFile, result);
        int unreadCount = numberAsInt(notificationCenter.getOrDefault("unreadCount", 0));
        int errorCount = numberAsInt(notificationCenter.getOrDefault("errorCount", 0));
        Map<String, Object> badge = new LinkedHashMap<>();
        badge.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge");
        badge.put("uploadFilesChecksumsNotificationPanelNotificationBadgeVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-notification-badge/v1");
        badge.put("label", unreadCount + " panel notifications");
        badge.put("status", notificationCenter.getOrDefault("status", "normal"));
        badge.put("tone", errorCount > 0 ? "critical" : "informational");
        badge.put("visible", unreadCount > 0);
        badge.put("unreadCount", unreadCount);
        badge.put("errorCount", errorCount);
        badge.put("notificationCount", notificationCenter.getOrDefault("notificationCount", 0));
        badge.put("activeNotificationId", notificationCenter.getOrDefault("activeNotificationId", ""));
        badge.put("notificationCenterSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationCenter:ci");
        badge.put("toastStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelToastState:ci");
        badge.put("viewStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelViewState:ci");
        badge.put("errorStateSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelErrorState:ci");
        badge.put("notificationCenter", notificationCenter);
        return badge;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine(String resultFile, Map<String, Object> result) {
        Object badgeObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadgeCi");
        Map<String, Object> badge = badgeObject instanceof Map ? asMap((Map<?, ?>) badgeObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(resultFile, result);
        return "checklistDashboardGateUploadFilesChecksumsNotificationPanelNotification"
                + " status=" + badge.getOrDefault("status", "normal")
                + " tone=" + badge.getOrDefault("tone", "informational")
                + " visible=" + badge.getOrDefault("visible", false)
                + " unreadCount=" + badge.getOrDefault("unreadCount", 0)
                + " errorCount=" + badge.getOrDefault("errorCount", 0)
                + " notificationCount=" + badge.getOrDefault("notificationCount", 0)
                + " activeNotificationId=" + badge.getOrDefault("activeNotificationId", "")
                + " notificationCenterSelector=" + badge.getOrDefault("notificationCenterSelector", "")
                + " badgeSelector=resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge:ci";
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine(String resultFile, Map<String, Object> result) {
        Object badgeObject = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadgeCi");
        Map<String, Object> badge = badgeObject instanceof Map ? asMap((Map<?, ?>) badgeObject) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge(resultFile, result);
        Map<String, Object> jsonLine = new LinkedHashMap<>();
        jsonLine.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationJsonLine");
        jsonLine.put("uploadFilesChecksumsNotificationPanelNotificationJsonLineVersion", "checklist-dashboard-gate-upload-files-checksums-notification-panel-notification-json-line/v1");
        jsonLine.put("status", badge.getOrDefault("status", "normal"));
        jsonLine.put("tone", badge.getOrDefault("tone", "informational"));
        jsonLine.put("visible", badge.getOrDefault("visible", false));
        jsonLine.put("unreadCount", badge.getOrDefault("unreadCount", 0));
        jsonLine.put("errorCount", badge.getOrDefault("errorCount", 0));
        jsonLine.put("notificationCount", badge.getOrDefault("notificationCount", 0));
        jsonLine.put("activeNotificationId", badge.getOrDefault("activeNotificationId", ""));
        jsonLine.put("notificationCenterSelector", badge.getOrDefault("notificationCenterSelector", ""));
        jsonLine.put("badgeSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationBadge:ci");
        jsonLine.put("lineSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsNotificationPanelNotificationLine:ci");
        try {
            return JSON_LINE.writeValueAsString(jsonLine);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to render gate upload checksum notification-panel notification JSON line.", exception);
        }
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        Map<String, Object> checksums = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile, result);
        Map<String, Object> jsonLine = new LinkedHashMap<>();
        jsonLine.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsJsonLine");
        jsonLine.put("uploadFilesChecksumsJsonLineVersion", "checklist-dashboard-gate-upload-files-checksums-json-line/v1");
        jsonLine.put("algorithm", checksums.getOrDefault("algorithm", "SHA-256"));
        jsonLine.put("fileCount", checksums.getOrDefault("fileCount", 0));
        jsonLine.put("files", checksums.getOrDefault("files", List.of()));
        jsonLine.put("checksumsSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        try {
            return JSON_LINE.writeValueAsString(jsonLine);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to render gate upload checksum JSON line.", exception);
        }
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        Map<String, Object> checksums = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile, result);
        Object filesObject = checksums.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> pairs = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            String sha256 = String.valueOf(file.getOrDefault("sha256", ""));
            if (!filename.isBlank() && !sha256.isBlank()) {
                pairs.add(filename + "=" + sha256);
            }
        }
        List<String> headers = List.of("summary", "csvLineVersion", "algorithm", "fileCount", "checksums", "checksumsSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCsvLine",
                "checklist-dashboard-gate-upload-files-checksums-csv-line/v1",
                String.valueOf(checksums.getOrDefault("algorithm", "SHA-256")),
                String.valueOf(checksums.getOrDefault("fileCount", pairs.size())),
                String.join(",", pairs),
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        return String.join(",", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList()) + "\n"
                + String.join(",", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        Map<String, Object> checksums = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile, result);
        Object filesObject = checksums.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> pairs = new ArrayList<>();
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            String sha256 = String.valueOf(file.getOrDefault("sha256", ""));
            if (!filename.isBlank() && !sha256.isBlank()) {
                pairs.add(filename + "=" + sha256);
            }
        }
        List<String> headers = List.of("summary", "tsvLineVersion", "algorithm", "fileCount", "checksums", "checksumsSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsTsvLine",
                "checklist-dashboard-gate-upload-files-checksums-tsv-line/v1",
                String.valueOf(checksums.getOrDefault("algorithm", "SHA-256")),
                String.valueOf(checksums.getOrDefault("fileCount", pairs.size())),
                String.join(",", pairs),
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci");
        return String.join("\t", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList()) + "\n"
                + String.join("\t", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsMarkdown(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        Map<String, Object> checksums = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile, result);
        Object filesObject = checksums.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> lines = new ArrayList<>();
        lines.add("**Checklist dashboard gate upload checksums**");
        lines.add("");
        lines.add("Algorithm: `" + escapeMarkdownTableCode(String.valueOf(checksums.getOrDefault("algorithm", "SHA-256"))) + "`");
        lines.add("");
        lines.add("| Filename | Bytes | SHA-256 |");
        lines.add("| --- | ---: | --- |");
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            lines.add("| " + escapeMarkdownTableCell(String.valueOf(file.getOrDefault("filename", "")))
                    + " | " + escapeMarkdownTableCell(String.valueOf(file.getOrDefault("byteLength", "")))
                    + " | `" + escapeMarkdownTableCode(String.valueOf(file.getOrDefault("sha256", ""))) + "` |");
        }
        lines.add("");
        lines.add("Checksums selector: `resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci`");
        return String.join("\n", lines);
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsEnv(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksumsCi");
        Map<String, Object> checksums = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums(resultFile, result);
        Object filesObject = checksums.get("files");
        List<?> files = filesObject instanceof List ? (List<?>) filesObject : List.of();
        List<String> filenames = new ArrayList<>();
        List<String> lines = new ArrayList<>();
        lines.add("GATE_UPLOAD_CHECKSUMS_ENV_VERSION=" + shellQuote("checklist-dashboard-gate-upload-files-checksums-env/v1"));
        lines.add("GATE_UPLOAD_CHECKSUM_ALGORITHM=" + shellQuote(String.valueOf(checksums.getOrDefault("algorithm", "SHA-256"))));
        lines.add("GATE_UPLOAD_CHECKSUM_FILE_COUNT=" + checksums.getOrDefault("fileCount", files.size()));
        lines.add("GATE_UPLOAD_CHECKSUMS_SELECTOR=" + shellQuote("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadFilesChecksums:ci"));
        int index = 1;
        for (Object fileObject : files) {
            if (!(fileObject instanceof Map)) {
                continue;
            }
            Map<String, Object> file = asMap((Map<?, ?>) fileObject);
            String filename = String.valueOf(file.getOrDefault("filename", ""));
            String sha256 = String.valueOf(file.getOrDefault("sha256", ""));
            if (filename.isBlank() || sha256.isBlank()) {
                continue;
            }
            filenames.add(filename);
            lines.add("GATE_UPLOAD_CHECKSUM_FILE_" + index + "=" + shellQuote(filename));
            lines.add("GATE_UPLOAD_CHECKSUM_SHA256_" + index + "=" + shellQuote(sha256));
            lines.add("GATE_UPLOAD_CHECKSUM_BYTES_" + index + "=" + file.getOrDefault("byteLength", 0));
            index++;
        }
        lines.add("GATE_UPLOAD_CHECKSUM_FILES=" + shellQuote(String.join(",", filenames)));
        return String.join("\n", lines);
    }

    private static String renderGateUploadFileContent(String resultFile, Map<String, Object> result, String selector, String contentType) {
        Object value;
        switch (selector) {
            case "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifact:ci":
                value = result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateArtifactSummary(resultFile, result));
                break;
            case "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummary:ci":
                value = result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateSummaryBundleSummary(resultFile, result));
                break;
            case "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJson:ci":
                value = result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateJsonSummary(resultFile, result));
                break;
            case "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLine:ci":
                value = result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateLineSummary(resultFile, result));
                break;
            case "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnv:ci":
                value = result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateEnvSummary(resultFile, result));
                break;
            case "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdown:ci":
                value = result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateMarkdownSummary(resultFile, result));
                break;
            default:
                value = result.get(selector.replace(":ci", "Ci"));
                break;
        }
        return "application/json".equals(contentType) ? renderValue(value, "json") : renderText(value);
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus(String resultFile, Map<String, Object> result) {
        Map<String, Object> uploadStatus = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile, result));
        uploadStatus.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        uploadStatus.put("resultFile", resultFile);
        return uploadStatus;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(String resultFile) {
        return resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile, new LinkedHashMap<>());
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestCi");
        Map<String, Object> manifest = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifestSummary(resultFile, result);
        Object filesObject = manifest.get("files");
        int fileCount = filesObject instanceof List ? ((List<?>) filesObject).size() : checklistDashboardGateUploadFiles().size();
        int selectorCount = checklistDashboardGateUploadIndexRows(resultFile).size();
        boolean ready = fileCount >= 6 && selectorCount >= 5;
        String status = ready ? "ready" : "incomplete";
        String statusLine = "checklistDashboardGateUploadStatus status=" + status + " ready=" + ready + " fileCount=" + fileCount + " selectorCount=" + selectorCount;

        Map<String, Object> uploadStatus = new LinkedHashMap<>();
        uploadStatus.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatus");
        uploadStatus.put("uploadStatusVersion", "checklist-dashboard-gate-upload-status/v1");
        uploadStatus.put("status", status);
        uploadStatus.put("ready", ready);
        uploadStatus.put("fileCount", fileCount);
        uploadStatus.put("selectorCount", selectorCount);
        uploadStatus.put("statusLine", statusLine);
        uploadStatus.put("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci");
        uploadStatus.put("indexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci");
        uploadStatus.put("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci");
        return uploadStatus;
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi");
        Map<String, Object> uploadStatus = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile, result);
        return String.valueOf(uploadStatus.getOrDefault("statusLine", "checklistDashboardGateUploadStatus status=unknown ready=false fileCount=0 selectorCount=0"));
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadEnv(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi");
        Map<String, Object> uploadStatus = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile, result);
        String ready = Boolean.TRUE.equals(uploadStatus.get("ready")) ? "true" : "false";
        return "GATE_UPLOAD_READY=" + ready + "\n"
                + "GATE_UPLOAD_STATUS=" + shellQuote(String.valueOf(uploadStatus.getOrDefault("status", "unknown"))) + "\n"
                + "GATE_UPLOAD_FILE_COUNT=" + uploadStatus.getOrDefault("fileCount", 0) + "\n"
                + "GATE_UPLOAD_SELECTOR_COUNT=" + uploadStatus.getOrDefault("selectorCount", 0) + "\n"
                + "GATE_UPLOAD_STATUS_LINE=" + shellQuote(String.valueOf(uploadStatus.getOrDefault("statusLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(resultFile, result)))) + "\n"
                + "GATE_UPLOAD_MANIFEST_SELECTOR=" + shellQuote(String.valueOf(uploadStatus.getOrDefault("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci"))) + "\n"
                + "GATE_UPLOAD_INDEX_SELECTOR=" + shellQuote(String.valueOf(uploadStatus.getOrDefault("indexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci"))) + "\n"
                + "GATE_UPLOAD_BUNDLE_SELECTOR=" + shellQuote(String.valueOf(uploadStatus.getOrDefault("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci")));
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi");
        Map<String, Object> uploadStatus = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile, result);
        Map<String, Object> jsonLine = new LinkedHashMap<>();
        jsonLine.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadJsonLine");
        jsonLine.put("uploadJsonLineVersion", "checklist-dashboard-gate-upload-json-line/v1");
        jsonLine.put("status", uploadStatus.getOrDefault("status", "unknown"));
        jsonLine.put("ready", uploadStatus.getOrDefault("ready", false));
        jsonLine.put("fileCount", uploadStatus.getOrDefault("fileCount", 0));
        jsonLine.put("selectorCount", uploadStatus.getOrDefault("selectorCount", 0));
        jsonLine.put("statusLine", uploadStatus.getOrDefault("statusLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(resultFile, result)));
        jsonLine.put("manifestSelector", uploadStatus.getOrDefault("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci"));
        jsonLine.put("indexSelector", uploadStatus.getOrDefault("indexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci"));
        jsonLine.put("bundleSelector", uploadStatus.getOrDefault("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci"));
        try {
            return JSON_LINE.writeValueAsString(jsonLine);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to render gate upload JSON line.", exception);
        }
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi");
        Map<String, Object> uploadStatus = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile, result);
        List<String> headers = List.of("summary", "csvLineVersion", "status", "ready", "fileCount", "selectorCount", "statusLine", "manifestSelector", "indexSelector", "bundleSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadCsvLine",
                "checklist-dashboard-gate-upload-csv-line/v1",
                String.valueOf(uploadStatus.getOrDefault("status", "unknown")),
                String.valueOf(uploadStatus.getOrDefault("ready", false)),
                String.valueOf(uploadStatus.getOrDefault("fileCount", 0)),
                String.valueOf(uploadStatus.getOrDefault("selectorCount", 0)),
                String.valueOf(uploadStatus.getOrDefault("statusLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(resultFile, result))),
                String.valueOf(uploadStatus.getOrDefault("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci")),
                String.valueOf(uploadStatus.getOrDefault("indexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci")),
                String.valueOf(uploadStatus.getOrDefault("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci")));
        return String.join(",", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList()) + "\n"
                + String.join(",", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList());
    }

    private static String resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine(String resultFile, Map<String, Object> result) {
        Object fromResult = result.get("resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusCi");
        Map<String, Object> uploadStatus = fromResult instanceof Map ? asMap((Map<?, ?>) fromResult) : resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusSummary(resultFile, result);
        List<String> headers = List.of("summary", "tsvLineVersion", "status", "ready", "fileCount", "selectorCount", "statusLine", "manifestSelector", "indexSelector", "bundleSelector");
        List<String> values = List.of(
                "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadTsvLine",
                "checklist-dashboard-gate-upload-tsv-line/v1",
                String.valueOf(uploadStatus.getOrDefault("status", "unknown")),
                String.valueOf(uploadStatus.getOrDefault("ready", false)),
                String.valueOf(uploadStatus.getOrDefault("fileCount", 0)),
                String.valueOf(uploadStatus.getOrDefault("selectorCount", 0)),
                String.valueOf(uploadStatus.getOrDefault("statusLine", resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadStatusLine(resultFile, result))),
                String.valueOf(uploadStatus.getOrDefault("manifestSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadManifest:ci")),
                String.valueOf(uploadStatus.getOrDefault("indexSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadIndex:ci")),
                String.valueOf(uploadStatus.getOrDefault("bundleSelector", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDashboardGateUploadBundle:ci")));
        return String.join("\t", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList()) + "\n"
                + String.join("\t", values.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeTsv).toList());
    }

    private static Map<String, Object> checklistAction(String id, String label, String kind, String tone, String selector) {
        Map<String, Object> action = new LinkedHashMap<>();
        action.put("id", id);
        action.put("label", label);
        action.put("kind", kind);
        action.put("tone", tone);
        action.put("selector", selector);
        return action;
    }

    private static Map<String, Object> checksumMenuGroup(String id, String label, List<Map<String, Object>> items) {
        Map<String, Object> group = new LinkedHashMap<>();
        group.put("id", id);
        group.put("label", label);
        group.put("itemCount", items.size());
        group.put("items", items);
        return group;
    }

    private static Map<String, Object> checksumMenuItem(String id, String label, String selector) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("id", id);
        item.put("label", label);
        item.put("selector", selector);
        return item;
    }

    private static Map<String, Object> checksumToolbarControl(String id, String label, String type, String kind, String selector) {
        Map<String, Object> control = new LinkedHashMap<>();
        control.put("id", id);
        control.put("label", label);
        control.put("type", type);
        control.put("kind", kind);
        control.put("selector", selector);
        return control;
    }

    private static Map<String, Object> checksumCommand(String id, String title, String group, String selector, String... keywords) {
        Map<String, Object> command = new LinkedHashMap<>();
        command.put("id", id);
        command.put("title", title);
        command.put("group", group);
        command.put("selector", selector);
        command.put("keywords", List.of(keywords));
        return command;
    }

    private static Map<String, Object> checksumSearchEntry(Map<String, Object> command) {
        Object keywords = command.getOrDefault("keywords", List.of());
        String keywordText = keywords instanceof List ? ((List<?>) keywords).stream().map(String::valueOf).collect(Collectors.joining(" ")) : String.valueOf(keywords);
        String text = String.join(" ",
                String.valueOf(command.getOrDefault("id", "")),
                String.valueOf(command.getOrDefault("title", "")),
                String.valueOf(command.getOrDefault("group", "")),
                String.valueOf(command.getOrDefault("selector", "")),
                keywordText).toLowerCase(Locale.ROOT);
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("id", command.getOrDefault("id", ""));
        entry.put("title", command.getOrDefault("title", ""));
        entry.put("group", command.getOrDefault("group", ""));
        entry.put("selector", command.getOrDefault("selector", ""));
        entry.put("keywords", keywords);
        entry.put("text", text);
        return entry;
    }

    private static Map<String, Integer> checksumGroupCounts(Object entries) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        if (entries instanceof List) {
            for (Object entryObject : (List<?>) entries) {
                if (entryObject instanceof Map) {
                    Map<String, Object> entry = asMap((Map<?, ?>) entryObject);
                    String group = String.valueOf(entry.getOrDefault("group", "unknown"));
                    counts.put(group, counts.getOrDefault(group, 0) + 1);
                }
            }
        }
        return counts;
    }

    private static Map<String, Object> checksumEmptyAction(String id, String label, String selector) {
        Map<String, Object> action = new LinkedHashMap<>();
        action.put("id", id);
        action.put("label", label);
        action.put("selector", selector);
        return action;
    }

    private static Map<String, Object> checksumLoadingSkeletonRow(String id, String label) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("id", id);
        row.put("label", label);
        row.put("state", "loading");
        return row;
    }

    private static Map<String, Object> checksumStateAction(String id, String label, String selector) {
        Map<String, Object> action = new LinkedHashMap<>();
        action.put("id", id);
        action.put("label", label);
        action.put("selector", selector);
        return action;
    }

    private static Map<String, Object> checksumToast(String id, String tone, String title, String body, String selector) {
        Map<String, Object> toast = new LinkedHashMap<>();
        toast.put("id", id);
        toast.put("tone", tone);
        toast.put("title", title);
        toast.put("body", body);
        toast.put("selector", selector);
        toast.put("dismissible", true);
        return toast;
    }

    private static Map<String, Object> checksumNotificationGroup(String id, String label, int count) {
        Map<String, Object> group = new LinkedHashMap<>();
        group.put("id", id);
        group.put("label", label);
        group.put("count", count);
        return group;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails(String resultFile, Map<String, Object> result) {
        Map<String, Object> details = new LinkedHashMap<>(resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsSummary(resultFile));
        details.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        details.put("resultFile", resultFile);
        details.put("checklist", result.getOrDefault("resultDashboardTreeCatalogTreeSummaryRegistryChecklistCi", resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile)));
        details.put("checks", resultDashboardTreeCatalogTreeSummaryRegistryChecklistRows(resultFile));
        return details;
    }

    private static Map<String, Object> resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetailsSummary(String resultFile) {
        List<Map<String, Object>> checks = resultDashboardTreeCatalogTreeSummaryRegistryChecklistRows(resultFile);
        Map<String, Object> checklist = resultDashboardTreeCatalogTreeSummaryRegistryChecklistSummary(resultFile);
        ArrayList arrayList = new ArrayList();
        for (Map<String, Object> row : checks) {
            arrayList.add(String.valueOf(row.getOrDefault("name", "")));
        }
        Map<String, Object> details = new LinkedHashMap<>();
        details.put("summary", "resultDashboardTreeCatalogTreeSummaryRegistryChecklistDetails");
        details.put("current", checklist.getOrDefault("current", "resultDashboardTreeCatalogTreeSummaryRegistry:ci"));
        details.put("reportStatus", checklist.getOrDefault("reportStatus", "NEEDS_ATTENTION"));
        details.put("checkCount", checklist.getOrDefault("checkCount", Integer.valueOf(checks.size())));
        details.put("passedCount", checklist.getOrDefault("passedCount", 0));
        details.put("failedCount", checklist.getOrDefault("failedCount", Integer.valueOf(checks.size())));
        details.put("allPassed", checklist.getOrDefault("allPassed", false));
        details.put("checks", arrayList);
        details.put("breadcrumb", checklist.getOrDefault("breadcrumb", ""));
        return details;
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogFieldRows() {
        return List.of(compareInspectFieldRow("paths", "object", "Compact tree path summary block."), compareInspectFieldRow("children", "object", "Compact tree child summary block."), compareInspectFieldRow("counts", "object", "Compact tree count summary block."));
    }

    private static Map<String, Object> resultDashboardTreeCatalogFieldSummary() {
        return Map.of("summary", "resultDashboardTreeCatalogFields", "fieldCount", 3, "fields", List.of("paths", "children", "counts"));
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogCommandRows(String resultFile) {
        return List.of(resultAliasRow("catalog", "resultDashboardTreeCatalog:ci", resultFile), resultAliasRow("catalogFields", "resultDashboardTreeCatalogFields:ci", resultFile), resultAliasRow("paths", "resultDashboardTreePaths:ci", resultFile), resultAliasRow("children", "resultDashboardTreeChildren:ci", resultFile), resultAliasRow("counts", "resultDashboardTreeCounts:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogCommandSummary(String resultFile) {
        List<String> commands = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogCommandRows(resultFile)) {
            commands.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogCommands", "commandCount", Integer.valueOf(commands.size()), "commands", commands);
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogSelectorRows(String resultFile) {
        return List.of(resultAliasRow("catalog", "resultDashboardTreeCatalog:ci", resultFile), resultAliasRow("catalogFields", "resultDashboardTreeCatalogFields:ci", resultFile), resultAliasRow("paths", "resultDashboardTreePaths:ci", resultFile), resultAliasRow("children", "resultDashboardTreeChildren:ci", resultFile), resultAliasRow("counts", "resultDashboardTreeCounts:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogSelectorSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogSelectorRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogSelectors", "selectorCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static List<Map<String, Object>> resultDashboardTreeCatalogAliasRows(String resultFile) {
        return List.of(resultAliasRow("catalog", "resultDashboardTreeCatalog:ci", resultFile), resultAliasRow("catalogFields", "resultDashboardTreeCatalogFields:ci", resultFile), resultAliasRow("catalogCommands", "resultDashboardTreeCatalogCommands:ci", resultFile), resultAliasRow("catalogSelectors", "resultDashboardTreeCatalogSelectors:ci", resultFile), resultAliasRow("catalogChildren", "resultDashboardTreeChildren:ci", resultFile), resultAliasRow("catalogCounts", "resultDashboardTreeCounts:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCatalogAliasSummary(String resultFile) {
        List<String> aliases = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCatalogAliasRows(resultFile)) {
            aliases.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "resultDashboardTreeCatalogAliases", "aliasCount", Integer.valueOf(aliases.size()), "aliases", aliases);
    }

    private static List<Map<String, Object>> resultDashboardTreeFieldRows() {
        return List.of(compareInspectFieldRow("dashboard", "object", "Embedded dashboard summary subtree."), compareInspectFieldRow("aliases", "object", "Compact alias summary used by the tree."), compareInspectFieldRow("fields", "object", "Compact dashboard-family field summary used by the tree."), compareInspectFieldRow("commands", "object", "Compact dashboard-family command summary used by the tree."), compareInspectFieldRow("selectors", "object", "Compact dashboard-family selector summary used by the tree."));
    }

    private static Map<String, Object> resultDashboardTreeFieldSummary() {
        return Map.of("summary", "resultDashboardTreeFields", "fieldCount", 5, "fields", List.of("dashboard", "aliases", "fields", "commands", "selectors"));
    }

    private static List<Map<String, Object>> resultDashboardTreeChildRows() {
        return List.of(compareInspectFieldRow("dashboard", "child", "Primary dashboard subtree with manifest, summary, overview, status, and health."), compareInspectFieldRow("aliases", "child", "Alias discovery subtree referenced by the dashboard tree."), compareInspectFieldRow("fields", "child", "Field discovery subtree referenced by the dashboard tree."), compareInspectFieldRow("commands", "child", "Command discovery subtree referenced by the dashboard tree."), compareInspectFieldRow("selectors", "child", "Selector discovery subtree referenced by the dashboard tree."));
    }

    private static Map<String, Object> resultDashboardTreeChildSummary() {
        return Map.of("summary", "resultDashboardTreeChildren", "childCount", 5, "children", List.of("dashboard", "aliases", "fields", "commands", "selectors"));
    }

    private static Map<String, Object> resultDashboardTreeCountSummary(String resultFile, Map<String, Object> result) {
        return Map.of("summary", "resultDashboardTreeCounts", "childCount", 5, "aliasCount", asMap(result.getOrDefault("resultDashboardAliasesCi", resultDashboardAliasSummary())).getOrDefault("aliasCount", 0), "fieldCount", asMap(result.getOrDefault("resultDashboardFamilyFieldsCi", resultDashboardFamilyFieldSummary())).getOrDefault("fieldCount", 0), "commandCount", asMap(result.getOrDefault("resultDashboardFamilyCommandsCi", resultDashboardFamilyCommandSummary(resultFile))).getOrDefault("commandCount", 0), "selectorCount", asMap(result.getOrDefault("resultDashboardFamilySelectorsCi", resultDashboardFamilySelectorSummary(resultFile))).getOrDefault("selectorCount", 0));
    }

    private static String resultDashboardTreeCountLine(Map<String, Object> summary) {
        return "resultDashboardTreeCounts childCount=" + String.valueOf(summary.getOrDefault("childCount", 0)) + " aliasCount=" + String.valueOf(summary.getOrDefault("aliasCount", 0)) + " fieldCount=" + String.valueOf(summary.getOrDefault("fieldCount", 0)) + " commandCount=" + String.valueOf(summary.getOrDefault("commandCount", 0)) + " selectorCount=" + String.valueOf(summary.getOrDefault("selectorCount", 0));
    }

    private static List<Map<String, Object>> resultDashboardTreeAliasRows() {
        return List.of(compareInspectFieldRow("resultDashboardTree", "alias", "Preferred dashboard-tree aggregate alias."), compareInspectFieldRow("resultDashboardTreeFields", "alias", "Preferred dashboard-tree field-summary alias."), compareInspectFieldRow("resultDashboardTreeCommands", "alias", "Preferred dashboard-tree command-summary alias."), compareInspectFieldRow("resultDashboardTreeSelectors", "alias", "Preferred dashboard-tree selector-summary alias."), compareInspectFieldRow("resultDashboardAliases", "alias", "Preferred dashboard alias-discovery family referenced by the tree."));
    }

    private static Map<String, Object> resultDashboardTreeAliasSummary() {
        return Map.of("summary", "short", "aliasCount", Integer.valueOf(resultDashboardTreeAliasRows().size()), "aliases", resultDashboardTreeAliasRows().stream().map(row -> {
            return row.get("name");
        }).toList());
    }

    private static List<Map<String, Object>> resultDashboardTreeCommandRows(String resultFile) {
        return List.of(resultAliasRow("tree", "resultDashboardTree:ci", resultFile), resultAliasRow("treeFields", "resultDashboardTreeFields:ci", resultFile), resultAliasRow("dashboardManifest", "resultDashboardManifest:ci", resultFile), resultAliasRow("dashboardSummary", "resultDashboardSummary:ci", resultFile), resultAliasRow("dashboardOverview", "resultDashboardOverview:ci", resultFile), resultAliasRow("dashboardStatus", "resultDashboardStatus:ci", resultFile), resultAliasRow("dashboardHealth", "resultDashboardHealth:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeCommandSummary(String resultFile) {
        List<String> commands = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeCommandRows(resultFile)) {
            commands.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "short", "commandCount", Integer.valueOf(commands.size()), "commands", commands);
    }

    private static List<Map<String, Object>> resultDashboardTreeSelectorRows(String resultFile) {
        return List.of(resultAliasRow("tree", "resultDashboardTree:ci", resultFile), resultAliasRow("treeFields", "resultDashboardTreeFields:ci", resultFile), resultAliasRow("dashboardManifest", "resultDashboardManifest:ci", resultFile), resultAliasRow("dashboardSummary", "resultDashboardSummary:ci", resultFile), resultAliasRow("dashboardOverview", "resultDashboardOverview:ci", resultFile), resultAliasRow("dashboardStatus", "resultDashboardStatus:ci", resultFile), resultAliasRow("dashboardHealth", "resultDashboardHealth:ci", resultFile));
    }

    private static Map<String, Object> resultDashboardTreeSelectorSummary(String resultFile) {
        List<String> selectors = new ArrayList<>();
        for (Map<String, Object> row : resultDashboardTreeSelectorRows(resultFile)) {
            selectors.add(String.valueOf(row.getOrDefault("selector", "")));
        }
        return Map.of("summary", "short", "selectorCount", Integer.valueOf(selectors.size()), "selectors", selectors);
    }

    private static Map<String, Object> resultDashboardSummary(String resultFile, Map<String, Object> result) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "dashboardSummary");
        summary.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        summary.put("resultFile", resultFile);
        summary.put("resultDashboardCi", result.getOrDefault("resultDashboardCi", resultDashboardLine(resultDashboard(resultFile, result))));
        summary.put("resultDashboardOverviewCi", result.getOrDefault("resultDashboardOverviewCi", resultDashboardOverviewLine(resultDashboardOverview(resultFile, result))));
        summary.put("resultDashboardStatusCi", result.getOrDefault("resultDashboardStatusCi", resultDashboardStatusLine(resultDashboardStatus(resultFile, result))));
        summary.put("resultDashboardHealthCi", result.getOrDefault("resultDashboardHealthCi", resultDashboardHealthLine(resultDashboardHealth(resultFile, result))));
        return summary;
    }

    private static String resultDashboardSummaryLine(Map<String, Object> summary) {
        return "resultDashboardSummary reportVersion=" + String.valueOf(summary.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(summary.getOrDefault("resultFile", "")) + " resultDashboardOverviewCi=" + String.valueOf(summary.getOrDefault("resultDashboardOverviewCi", "")) + " resultDashboardStatusCi=" + String.valueOf(summary.getOrDefault("resultDashboardStatusCi", "")) + " resultDashboardHealthCi=" + String.valueOf(summary.getOrDefault("resultDashboardHealthCi", ""));
    }

    private static Map<String, Object> resultDashboardOverview(String resultFile, Map<String, Object> result) {
        Map<String, Object> overview = new LinkedHashMap<>();
        overview.put("summary", "dashboardOverview");
        overview.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        overview.put("resultFile", resultFile);
        overview.put("resultDashboardCi", result.getOrDefault("resultDashboardCi", resultDashboardLine(resultDashboard(resultFile, result))));
        overview.put("resultOverviewCi", result.getOrDefault("resultOverviewCi", ""));
        overview.put("resultComparisonCi", result.getOrDefault("resultComparisonCi", ""));
        overview.put("resultStatusCi", result.getOrDefault("resultStatusCi", ""));
        return overview;
    }

    private static String resultDashboardOverviewLine(Map<String, Object> overview) {
        return "resultDashboardOverview reportVersion=" + String.valueOf(overview.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(overview.getOrDefault("resultFile", "")) + " resultOverviewCi=" + String.valueOf(overview.getOrDefault("resultOverviewCi", "")) + " resultComparisonCi=" + String.valueOf(overview.getOrDefault("resultComparisonCi", "")) + " resultStatusCi=" + String.valueOf(overview.getOrDefault("resultStatusCi", ""));
    }

    private static Map<String, Object> resultDashboardStatus(String resultFile, Map<String, Object> result) {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("summary", "dashboardStatus");
        status.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        status.put("resultFile", resultFile);
        status.put("resultDashboardCi", result.getOrDefault("resultDashboardCi", resultDashboardLine(resultDashboard(resultFile, result))));
        status.put("resultStatusCi", result.getOrDefault("resultStatusCi", ""));
        status.put("resultHealthCi", result.getOrDefault("resultHealthCi", ""));
        status.put("resultComparisonCi", result.getOrDefault("resultComparisonCi", ""));
        return status;
    }

    private static String resultDashboardStatusLine(Map<String, Object> status) {
        return "resultDashboardStatus reportVersion=" + String.valueOf(status.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(status.getOrDefault("resultFile", "")) + " resultStatusCi=" + String.valueOf(status.getOrDefault("resultStatusCi", "")) + " resultHealthCi=" + String.valueOf(status.getOrDefault("resultHealthCi", "")) + " resultComparisonCi=" + String.valueOf(status.getOrDefault("resultComparisonCi", ""));
    }

    private static Map<String, Object> resultDashboardHealth(String resultFile, Map<String, Object> result) {
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("summary", "dashboardHealth");
        health.put("reportVersion", result.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION));
        health.put("resultFile", resultFile);
        health.put("resultDashboardCi", result.getOrDefault("resultDashboardCi", resultDashboardLine(resultDashboard(resultFile, result))));
        health.put("resultHealthCi", result.getOrDefault("resultHealthCi", ""));
        health.put("resultStatusCi", result.getOrDefault("resultStatusCi", ""));
        health.put("resultComparisonCi", result.getOrDefault("resultComparisonCi", ""));
        return health;
    }

    private static String resultDashboardHealthLine(Map<String, Object> health) {
        return "resultDashboardHealth reportVersion=" + String.valueOf(health.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(health.getOrDefault("resultFile", "")) + " resultHealthCi=" + String.valueOf(health.getOrDefault("resultHealthCi", "")) + " resultStatusCi=" + String.valueOf(health.getOrDefault("resultStatusCi", "")) + " resultComparisonCi=" + String.valueOf(health.getOrDefault("resultComparisonCi", ""));
    }

    private static String resultDashboardLine(Map<String, Object> dashboard) {
        return "resultDashboard reportVersion=" + String.valueOf(dashboard.getOrDefault("reportVersion", COMPARE_RESULT_REPORT_VERSION)) + " resultFile=" + String.valueOf(dashboard.getOrDefault("resultFile", "")) + " resultComparisonCi=" + String.valueOf(dashboard.getOrDefault("resultComparisonCi", "")) + " resultStatusCi=" + String.valueOf(dashboard.getOrDefault("resultStatusCi", "")) + " resultHealthCi=" + String.valueOf(dashboard.getOrDefault("resultHealthCi", ""));
    }

    private static List<Map<String, Object>> resultDashboardFieldRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(compareInspectFieldRow("reportVersion", "string", "Standalone compare-result schema marker"));
        rows.add(compareInspectFieldRow("resultFile", "string", "Standalone compare-result file path"));
        rows.add(compareInspectFieldRow("resultOverviewCi", "string", "Compact reopened-result overview summary"));
        rows.add(compareInspectFieldRow("resultComparisonCi", "string", "Compact reopened-result comparison summary"));
        rows.add(compareInspectFieldRow("resultSourcesCi", "string", "Compact reopened-result source summary"));
        rows.add(compareInspectFieldRow("resultStatusCi", "string", "Compact reopened-result status summary"));
        rows.add(compareInspectFieldRow("resultHealthCi", "string", "Compact reopened-result health summary"));
        rows.add(compareInspectFieldRow("resultAliasesCi", "object", "Compact reopened-result alias discovery summary"));
        rows.add(compareInspectFieldRow("resultSelectorsCi", "object", "Compact reopened-result selector discovery summary"));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultDashboardFieldSummary() {
        return Map.of("summary", "short", "fieldCount", Integer.valueOf(resultDashboardFieldRows().size()), "fields", resultDashboardFieldRows().stream().map(row -> {
            return row.get("name");
        }).toList());
    }

    private static List<Map<String, Object>> resultDashboardFamilyFieldRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(compareInspectFieldRow("resultDashboardManifest", "object", "Artifact-centric full dashboard payload alias"));
        rows.add(compareInspectFieldRow("resultDashboardSummary", "object", "Artifact-centric dashboard summary alias"));
        rows.add(compareInspectFieldRow("resultDashboardOverview", "object", "Artifact-centric dashboard overview alias"));
        rows.add(compareInspectFieldRow("resultDashboardStatus", "object", "Artifact-centric dashboard status alias"));
        rows.add(compareInspectFieldRow("resultDashboardHealth", "object", "Artifact-centric dashboard health alias"));
        rows.add(compareInspectFieldRow("resultDashboardAliases", "array", "Dashboard alias discovery rows"));
        rows.add(compareInspectFieldRow("resultDashboardFamilyCommands", "array", "Dashboard-family-only command rows"));
        rows.add(compareInspectFieldRow("resultDashboardFamilySelectors", "array", "Dashboard-family-only selector rows"));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultDashboardFamilyFieldSummary() {
        return Map.of("summary", "short", "fieldCount", Integer.valueOf(resultDashboardFamilyFieldRows().size()), "fields", resultDashboardFamilyFieldRows().stream().map(row -> {
            return row.get("name");
        }).toList());
    }

    private static List<Map<String, Object>> resultDashboardCommandRows(String resultFile) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(resultAliasRow("dashboard", "resultDashboard:ci", resultFile));
        rows.add(resultAliasRow("dashboardManifest", "resultDashboardManifest:ci", resultFile));
        rows.add(resultAliasRow("dashboardSummary", "resultDashboardSummary:ci", resultFile));
        rows.add(resultAliasRow("dashboardOverview", "resultDashboardOverview:ci", resultFile));
        rows.add(resultAliasRow("dashboardStatus", "resultDashboardStatus:ci", resultFile));
        rows.add(resultAliasRow("dashboardHealth", "resultDashboardHealth:ci", resultFile));
        rows.add(resultAliasRow("overview", "resultOverview:ci", resultFile));
        rows.add(resultAliasRow("comparison", "resultComparison:ci", resultFile));
        rows.add(resultAliasRow("sources", "resultSources:ci", resultFile));
        rows.add(resultAliasRow("status", "resultStatus:ci", resultFile));
        rows.add(resultAliasRow("health", "resultHealth:ci", resultFile));
        rows.add(resultAliasRow("aliases", "resultAliases:ci", resultFile));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultDashboardCommandSummary(String resultFile) {
        return Map.of("summary", "short", "commandCount", Integer.valueOf(resultDashboardCommandRows(resultFile).size()), "commands", resultDashboardCommandRows(resultFile).stream().map(row -> {
            return row.get("command");
        }).toList());
    }

    private static List<Map<String, Object>> resultDashboardFamilyCommandRows(String resultFile) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(resultAliasRow("dashboard", "resultDashboard:ci", resultFile));
        rows.add(resultAliasRow("dashboardManifest", "resultDashboardManifest:ci", resultFile));
        rows.add(resultAliasRow("dashboardSummary", "resultDashboardSummary:ci", resultFile));
        rows.add(resultAliasRow("dashboardOverview", "resultDashboardOverview:ci", resultFile));
        rows.add(resultAliasRow("dashboardStatus", "resultDashboardStatus:ci", resultFile));
        rows.add(resultAliasRow("dashboardHealth", "resultDashboardHealth:ci", resultFile));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultDashboardFamilyCommandSummary(String resultFile) {
        return Map.of("summary", "short", "commandCount", Integer.valueOf(resultDashboardFamilyCommandRows(resultFile).size()), "commands", resultDashboardFamilyCommandRows(resultFile).stream().map(row -> {
            return row.get("command");
        }).toList());
    }

    private static List<Map<String, Object>> resultDashboardAliasRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(compareInspectFieldRow("resultDashboardManifest", "alias", "Preferred reopened-result full dashboard payload alias family"));
        rows.add(compareInspectFieldRow("resultDashboardSummary", "alias", "Preferred reopened-result dashboard summary alias family"));
        rows.add(compareInspectFieldRow("resultDashboardOverview", "alias", "Preferred reopened-result dashboard overview alias family"));
        rows.add(compareInspectFieldRow("resultDashboardStatus", "alias", "Preferred reopened-result dashboard status alias family"));
        rows.add(compareInspectFieldRow("resultDashboardHealth", "alias", "Preferred reopened-result dashboard health alias family"));
        rows.add(compareInspectFieldRow("resultOverview", "alias", "Preferred reopened-result overview alias family"));
        rows.add(compareInspectFieldRow("resultComparison", "alias", "Preferred reopened-result comparison alias family"));
        rows.add(compareInspectFieldRow("resultSources", "alias", "Preferred reopened-result source alias family"));
        rows.add(compareInspectFieldRow("resultStatus", "alias", "Preferred reopened-result status alias family"));
        rows.add(compareInspectFieldRow("resultHealth", "alias", "Preferred reopened-result health alias family"));
        rows.add(compareInspectFieldRow("resultAliases", "alias", "Preferred reopened-result alias discovery family"));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultDashboardAliasSummary() {
        return Map.of("summary", "short", "aliasCount", Integer.valueOf(resultDashboardAliasRows().size()), "aliases", resultDashboardAliasRows().stream().map(row -> {
            return row.get("name");
        }).toList());
    }

    private static List<Map<String, Object>> resultDashboardSelectorRows(String resultFile) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(resultAliasRow("dashboard", "resultDashboard:ci", resultFile));
        rows.add(resultAliasRow("dashboardManifest", "resultDashboardManifest:ci", resultFile));
        rows.add(resultAliasRow("dashboardSummary", "resultDashboardSummary:ci", resultFile));
        rows.add(resultAliasRow("dashboardOverview", "resultDashboardOverview:ci", resultFile));
        rows.add(resultAliasRow("dashboardStatus", "resultDashboardStatus:ci", resultFile));
        rows.add(resultAliasRow("dashboardHealth", "resultDashboardHealth:ci", resultFile));
        rows.add(resultAliasRow("overview", "resultOverview:ci", resultFile));
        rows.add(resultAliasRow("comparison", "resultComparison:ci", resultFile));
        rows.add(resultAliasRow("sources", "resultSources:ci", resultFile));
        rows.add(resultAliasRow("status", "resultStatus:ci", resultFile));
        rows.add(resultAliasRow("health", "resultHealth:ci", resultFile));
        rows.add(resultAliasRow("aliases", "resultAliases:ci", resultFile));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultDashboardSelectorSummary(String resultFile) {
        return Map.of("summary", "short", "selectorCount", Integer.valueOf(resultDashboardSelectorRows(resultFile).size()), "selectors", resultDashboardSelectorRows(resultFile).stream().map(row -> {
            return row.get("selector");
        }).toList());
    }

    private static List<Map<String, Object>> resultDashboardFamilySelectorRows(String resultFile) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(resultAliasRow("dashboard", "resultDashboard:ci", resultFile));
        rows.add(resultAliasRow("dashboardManifest", "resultDashboardManifest:ci", resultFile));
        rows.add(resultAliasRow("dashboardSummary", "resultDashboardSummary:ci", resultFile));
        rows.add(resultAliasRow("dashboardOverview", "resultDashboardOverview:ci", resultFile));
        rows.add(resultAliasRow("dashboardStatus", "resultDashboardStatus:ci", resultFile));
        rows.add(resultAliasRow("dashboardHealth", "resultDashboardHealth:ci", resultFile));
        return List.copyOf(rows);
    }

    private static Map<String, Object> resultDashboardFamilySelectorSummary(String resultFile) {
        return Map.of("summary", "short", "selectorCount", Integer.valueOf(resultDashboardFamilySelectorRows(resultFile).size()), "selectors", resultDashboardFamilySelectorRows(resultFile).stream().map(row -> {
            return row.get("selector");
        }).toList());
    }

    private static Map<String, Object> bundleFileRow(String section, Path file) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("section", section);
        row.put("name", file.getFileName().toString());
        row.put("path", file.toString());
        return row;
    }

    private static Map<String, Object> summarizeBundle(BundleInput bundle) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("reportVersion", bundle.manifest().getOrDefault("reportVersion", COMPARE_BUNDLE_REPORT_VERSION));
        summary.put("bundleType", bundle.manifest().getOrDefault("bundleType", ""));
        summary.put("bundleMode", bundle.manifest().getOrDefault("bundleMode", ""));
        summary.put("bundleDir", bundle.bundleDir().toAbsolutePath().normalize().toString());
        summary.put("manifestFile", bundle.manifestFile().toAbsolutePath().normalize().toString());
        summary.put("leftReportFile", bundle.manifest().getOrDefault("leftReportFile", ""));
        summary.put("rightReportFile", bundle.manifest().getOrDefault("rightReportFile", ""));
        summary.put("leftReportVersion", bundle.manifest().getOrDefault("leftReportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        summary.put("rightReportVersion", bundle.manifest().getOrDefault("rightReportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        List<Map<String, Object>> files = bundleFiles(bundle);
        summary.put("fileCount", Integer.valueOf(files.size()));
        summary.put("sections", files.stream().map(row -> {
            return String.valueOf(row.getOrDefault("section", ""));
        }).toList());
        return summary;
    }

    private static Object summarizeBundleArtifactSummary(BundleInput bundle) {
        Object fromManifest = bundle.manifest().get("artifactSummary");
        if (fromManifest instanceof Map) {
            Map<?, ?> map = (Map) fromManifest;
            return asMap(map);
        }
        return bundleArtifactSummary(bundle.bundleDir().toAbsolutePath().normalize().toString(), bundle.manifest());
    }

    private static String summarizeBundleArtifactSummaryLine(BundleInput bundle) {
        Object fromManifest = bundle.manifest().get("artifactSummaryLine");
        if (fromManifest instanceof String) {
            String value = (String) fromManifest;
            if (!value.isBlank()) {
                return value;
            }
        }
        return bundleArtifactSummaryLine(asMap(summarizeBundleArtifactSummary(bundle)));
    }

    private static Map<String, Object> bundleArtifactSummary(String bundleDir, Map<String, Object> manifest) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "short");
        summary.put("bundleDir", bundleDir);
        summary.put("reportVersion", manifest.getOrDefault("reportVersion", COMPARE_BUNDLE_REPORT_VERSION));
        summary.put("comparisonSummary", manifest.getOrDefault("comparisonSummary", ""));
        summary.put("statusSummary", manifest.getOrDefault("statusSummary", ""));
        summary.put("commandsCi", manifest.getOrDefault("commandsCi", Map.of()));
        summary.put("selectorsCi", manifest.getOrDefault("selectorsCi", Map.of()));
        summary.put("inspectfieldsCi", manifest.getOrDefault("inspectfieldsCi", Map.of()));
        return summary;
    }

    private static String bundleArtifactSummaryLine(Map<String, Object> summary) {
        return "artifactSummary reportVersion=" + String.valueOf(summary.getOrDefault("reportVersion", COMPARE_BUNDLE_REPORT_VERSION)) + " bundleDir=" + String.valueOf(summary.getOrDefault("bundleDir", "")) + " comparisonSummary=" + String.valueOf(summary.getOrDefault("comparisonSummary", "")) + " statusSummary=" + String.valueOf(summary.getOrDefault("statusSummary", ""));
    }

    private static Map<String, Object> summarizeBundleManifestStatus(List<Map<String, Object>> files, Path bundleDir, Map<String, Object> manifest) {
        int missingFileCount = 0;
        for (Map<String, Object> row : files) {
            Path filePath = Path.of(String.valueOf(row.getOrDefault("path", "")), new String[0]);
            if (!Files.isRegularFile(filePath, new LinkOption[0])) {
                missingFileCount++;
            }
        }
        int fileCount = files.size();
        int okFileCount = fileCount - missingFileCount;
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("bundleType", manifest.getOrDefault("bundleType", ""));
        status.put("bundleDir", bundleDir.toAbsolutePath().normalize().toString());
        status.put("fileCount", Integer.valueOf(fileCount));
        status.put("okFileCount", Integer.valueOf(okFileCount));
        status.put("missingFileCount", Integer.valueOf(missingFileCount));
        status.put("healthStatus", missingFileCount == 0 ? "ok" : "missing-files");
        return status;
    }

    private static String bundleManifestStatusSummary(Map<String, Object> status) {
        return "status healthStatus=" + String.valueOf(status.getOrDefault("healthStatus", "")) + " fileCount=" + String.valueOf(status.getOrDefault("fileCount", 0)) + " okFileCount=" + String.valueOf(status.getOrDefault("okFileCount", 0)) + " missingFileCount=" + String.valueOf(status.getOrDefault("missingFileCount", 0));
    }

    private static Object summarizeBundleOverview(BundleInput bundle, Map<String, String> options) {
        String focus;
        if ("true".equals(options.get("ci")) || "short".equals(options.get("summary"))) {
            return summarizeBundleHealthShort(bundle);
        }
        focus = options.getOrDefault("focus", "all");
        switch (focus) {
            case "all":
                return summarizeBundle(bundle);
            case "health":
                return summarizeBundleHealthMap(bundle);
            case "files":
                return bundleFiles(bundle);
            default:
                throw new IllegalArgumentException("Unsupported bundleOverview focus '" + focus + "'. Use all, health, or files.");
        }
    }

    private static Map<String, Object> summarizeBundleHealth(BundleInput bundle) {
        return summarizeBundleHealthMap(bundle);
    }

    private static Object summarizeBundleHealth(BundleInput bundle, Map<String, String> options) {
        String focus;
        if ("true".equals(options.get("ci")) || "short".equals(options.get("summary"))) {
            return summarizeBundleHealthShort(bundle);
        }
        focus = options.getOrDefault("focus", "all");
        switch (focus) {
            case "all":
                return summarizeBundleHealthMap(bundle);
            case "files":
                return summarizeBundleHealthFiles(bundle, options);
            case "sections":
                return summarizeBundleHealthGrouped(bundle, "section", options);
            case "formats":
                return summarizeBundleHealthGrouped(bundle, "format", options);
            default:
                throw new IllegalArgumentException("Unsupported bundleHealth focus '" + focus + "'. Use files, sections, formats, or all.");
        }
    }

    private static Map<String, Object> summarizeBundleHealthMap(BundleInput bundle) {
        List<Map<String, Object>> files = bundleFiles(bundle);
        ArrayList arrayList = new ArrayList();
        for (Map<String, Object> row : files) {
            Path filePath = Path.of(String.valueOf(row.getOrDefault("path", "")), new String[0]);
            if (!Files.isRegularFile(filePath, new LinkOption[0])) {
                arrayList.add(filePath.toAbsolutePath().normalize().toString());
            }
        }
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("bundleType", bundle.manifest().getOrDefault("bundleType", ""));
        health.put("bundleDir", bundle.bundleDir().toAbsolutePath().normalize().toString());
        health.put("manifestFile", bundle.manifestFile().toAbsolutePath().normalize().toString());
        health.put("fileCount", Integer.valueOf(files.size()));
        health.put("missingFileCount", Integer.valueOf(arrayList.size()));
        health.put("missingFiles", arrayList);
        health.put("healthStatus", arrayList.isEmpty() ? "ok" : "missing-files");
        return health;
    }

    private static Map<String, Object> summarizeBundleHealthShort(BundleInput bundle) {
        Map<String, Object> health = summarizeBundleHealthMap(bundle);
        Map<String, Object> shortHealth = new LinkedHashMap<>();
        shortHealth.put("summary", "short");
        shortHealth.put("bundleDir", health.getOrDefault("bundleDir", ""));
        shortHealth.put("fileCount", health.getOrDefault("fileCount", 0));
        shortHealth.put("missingFileCount", health.getOrDefault("missingFileCount", 0));
        shortHealth.put("healthStatus", health.getOrDefault("healthStatus", ""));
        return shortHealth;
    }

    private static Object summarizeBundleCommands(BundleInput bundle, boolean ci) {
        String bundleDir = bundle.bundleDir().toAbsolutePath().normalize().toString();
        if (ci) {
            Object fromCommandsCi = bundle.manifest().get("commandsCi");
            if (fromCommandsCi instanceof Map) {
                Map<?, ?> map = (Map) fromCommandsCi;
                return asMap(map);
            }
            Object fromManifest = bundle.manifest().get("inspectCommandsCi");
            if (fromManifest instanceof Map) {
                Map<?, ?> map2 = (Map) fromManifest;
                return asMap(map2);
            }
            return bundleCommandSummary(bundleDir);
        }
        Object fromCommands = bundle.manifest().get("commands");
        if (fromCommands instanceof List) {
            List<?> list = (List) fromCommands;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map3 = (Map) item;
                    rows.add(asMap(map3));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        Object fromManifest2 = bundle.manifest().get("inspectCommands");
        if (fromManifest2 instanceof List) {
            List<?> list2 = (List) fromManifest2;
            List<Map<String, Object>> rows2 = new ArrayList<>();
            for (Object item2 : list2) {
                if (item2 instanceof Map) {
                    Map<?, ?> map4 = (Map) item2;
                    rows2.add(asMap(map4));
                }
            }
            if (!rows2.isEmpty()) {
                return List.copyOf(rows2);
            }
        }
        return bundleCommandRows(bundleDir);
    }

    private static Object summarizeBundleSelectors(BundleInput bundle, boolean ci) {
        String bundleDir = bundle.bundleDir().toAbsolutePath().normalize().toString();
        if (ci) {
            Object fromManifest = bundle.manifest().get("selectorsCi");
            if (fromManifest instanceof Map) {
                Map<?, ?> map = (Map) fromManifest;
                return asMap(map);
            }
            return bundleSelectorSummary(bundleDir);
        }
        Object fromManifest2 = bundle.manifest().get("selectors");
        if (fromManifest2 instanceof List) {
            List<?> list = (List) fromManifest2;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map2 = (Map) item;
                    rows.add(asMap(map2));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        return bundleSelectorRows(bundleDir);
    }

    private static List<Map<String, Object>> bundleCommandRows(String bundleDir) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(bundleCommandRow("status", "bundleStatus", bundleDir));
        rows.add(bundleCommandRow("health", "bundleHealth:ci", bundleDir));
        rows.add(bundleCommandRow("overview", "bundleOverview:ci", bundleDir));
        rows.add(bundleCommandRow("files", "files:section=runs:sort=name", bundleDir));
        rows.add(bundleCommandRow("nextTokens", "loadfile:delta-nextTokens", bundleDir));
        rows.add(bundleCommandRow("nextTokensPreview", "loadfile:delta-nextTokensPreview", bundleDir));
        rows.add(bundleCommandRow("nextTokensCount", "loadfile:delta-nextTokensCount", bundleDir));
        rows.add(bundleCommandRow("loadfile", "loadfile:section=runs:pick=last", bundleDir));
        return List.copyOf(rows);
    }

    private static Map<String, Object> bundleCommandSummary(String bundleDir) {
        List<String> parts = new ArrayList<>();
        for (Map<String, Object> row : bundleCommandRows(bundleDir)) {
            parts.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "short", "bundleDir", bundleDir, "commands", parts);
    }

    private static Map<String, Object> bundleCommandRow(String name, String selector, String bundleDir) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", name);
        row.put("selector", selector);
        row.put("command", bundleCommandText(selector, bundleDir));
        return row;
    }

    private static List<Map<String, Object>> bundleSelectorRows(String bundleDir) {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(bundleCommandRow("reportVersion", "reportVersion", bundleDir));
        rows.add(bundleCommandRow("comparison", "comparisonSummary", bundleDir));
        rows.add(bundleCommandRow("overview", "bundleOverview:ci", bundleDir));
        rows.add(bundleCommandRow("status", "bundleStatus", bundleDir));
        rows.add(bundleCommandRow("health", "bundleHealth:ci", bundleDir));
        rows.add(bundleCommandRow("files", "files:section=runs:sort=name", bundleDir));
        rows.add(bundleCommandRow("nextTokensPreview", "loadfile:delta-nextTokensPreview", bundleDir));
        return List.copyOf(rows);
    }

    private static Map<String, Object> bundleSelectorSummary(String bundleDir) {
        List<String> parts = new ArrayList<>();
        for (Map<String, Object> row : bundleSelectorRows(bundleDir)) {
            parts.add(String.valueOf(row.getOrDefault("command", "")));
        }
        return Map.of("summary", "short", "bundleDir", bundleDir, "selectors", parts);
    }

    private static String bundleCommandText(String selector, String bundleDir) {
        return "jbang trainer/trainer_byte_latent_train_infer_compare_inspector.java " + bundleDir + " " + selector;
    }

    private static String compareCommandText(String leftTarget, String rightTarget, String selector) {
        return "jbang trainer/trainer_byte_latent_train_infer_compare_inspector.java " + leftTarget + " " + rightTarget + " " + selector;
    }

    private static Object summarizeBundleInspectFields(BundleInput bundle, boolean ci) {
        if (ci) {
            Object fromTopLevel = bundle.manifest().get("inspectfieldsCi");
            if (fromTopLevel instanceof Map) {
                Map<?, ?> map = (Map) fromTopLevel;
                return asMap(map);
            }
            Object fromManifest = bundle.manifest().get("inspectFieldsCi");
            if (fromManifest instanceof Map) {
                Map<?, ?> map2 = (Map) fromManifest;
                return asMap(map2);
            }
            return bundleInspectFieldSummary();
        }
        Object fromTopLevel2 = bundle.manifest().get("inspectfields");
        if (fromTopLevel2 instanceof List) {
            List<?> list = (List) fromTopLevel2;
            List<Map<String, Object>> rows = new ArrayList<>();
            for (Object item : list) {
                if (item instanceof Map) {
                    Map<?, ?> map3 = (Map) item;
                    rows.add(asMap(map3));
                }
            }
            if (!rows.isEmpty()) {
                return List.copyOf(rows);
            }
        }
        Object fromManifest2 = bundle.manifest().get("inspectFields");
        if (fromManifest2 instanceof List) {
            List<?> list2 = (List) fromManifest2;
            List<Map<String, Object>> rows2 = new ArrayList<>();
            for (Object item2 : list2) {
                if (item2 instanceof Map) {
                    Map<?, ?> map4 = (Map) item2;
                    rows2.add(asMap(map4));
                }
            }
            if (!rows2.isEmpty()) {
                return List.copyOf(rows2);
            }
        }
        return bundleInspectFieldRows();
    }

    private static List<Map<String, Object>> bundleInspectFieldRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(bundleInspectFieldRow("reportVersion", "string", "Compare bundle manifest schema marker"));
        rows.add(bundleInspectFieldRow("bundleType", "string", "Compare bundle type identifier"));
        rows.add(bundleInspectFieldRow("bundleMode", "string", "Compare bundle export mode"));
        rows.add(bundleInspectFieldRow("bundleDir", "string", "Exported bundle directory path"));
        rows.add(bundleInspectFieldRow("leftReportFile", "string", "Left train-infer report path"));
        rows.add(bundleInspectFieldRow("rightReportFile", "string", "Right train-infer report path"));
        rows.add(bundleInspectFieldRow("leftReportVersion", "string", "Left train-infer report schema marker"));
        rows.add(bundleInspectFieldRow("rightReportVersion", "string", "Right train-infer report schema marker"));
        rows.add(bundleInspectFieldRow("fileCount", "number", "Number of generated bundle files"));
        rows.add(bundleInspectFieldRow("comparison", "object", "Compact compare summary block with preview and change signals"));
        rows.add(bundleInspectFieldRow("comparisonSummary", "string", "Human-readable one-line compare summary"));
        rows.add(bundleInspectFieldRow("artifactSummary", "object", "Compact compare-bundle artifact dashboard"));
        rows.add(bundleInspectFieldRow("artifactSummaryLine", "string", "Human-readable one-line compare-bundle artifact dashboard"));
        rows.add(bundleInspectFieldRow("leftNextTokensPreview", "array", "Top-level left compact next-tokens preview"));
        rows.add(bundleInspectFieldRow("rightNextTokensPreview", "array", "Top-level right compact next-tokens preview"));
        rows.add(bundleInspectFieldRow("nextTokensPreviewChanged", "boolean", "Whether the compact next-tokens preview changed"));
        rows.add(bundleInspectFieldRow("status", "object", "Compact manifest health status block"));
        rows.add(bundleInspectFieldRow("statusSummary", "string", "Human-readable one-line manifest status"));
        rows.add(bundleInspectFieldRow("healthStatus", "string", "Top-level bundle health status"));
        rows.add(bundleInspectFieldRow("missingFileCount", "number", "Top-level missing generated file count"));
        rows.add(bundleInspectFieldRow("okFileCount", "number", "Top-level generated file count that exists"));
        rows.add(bundleInspectFieldRow("files", "array", "Generated bundle file entries"));
        rows.add(bundleInspectFieldRow("inspectBundleStatus", "string", "Direct bundle status command"));
        rows.add(bundleInspectFieldRow("inspectBundleHealth", "string", "Direct bundle health command"));
        rows.add(bundleInspectFieldRow("inspectBundleOverview", "string", "Direct bundle overview command"));
        rows.add(bundleInspectFieldRow("inspectBundleFiles", "string", "Direct filtered bundle files command"));
        rows.add(bundleInspectFieldRow("inspectBundleLoadfile", "string", "Direct bundle loadfile command"));
        rows.add(bundleInspectFieldRow("inspectBundleSummarySelector", "string", "Recommended bundle summary selector"));
        rows.add(bundleInspectFieldRow("inspectBundleFilesSummarySelector", "string", "Recommended bundle file summary selector"));
        rows.add(bundleInspectFieldRow("inspectBundleNextTokensSelector", "string", "Recommended next-tokens delta selector"));
        rows.add(bundleInspectFieldRow("inspectBundleNextTokensPreviewSelector", "string", "Recommended next-tokens preview delta selector"));
        rows.add(bundleInspectFieldRow("inspectBundleNextTokensCountSelector", "string", "Recommended next-tokens-count delta selector"));
        rows.add(bundleInspectFieldRow("inspectBundleMissingFilesSelector", "string", "Recommended missing-files selector"));
        rows.add(bundleInspectFieldRow("inspectBundleMissingFilesTopSelector", "string", "Recommended trimmed missing-files selector"));
        rows.add(bundleInspectFieldRow("commands", "array", "Recommended top-level compare-bundle commands"));
        rows.add(bundleInspectFieldRow("commandsCi", "object", "Compact top-level compare-bundle command summary"));
        rows.add(bundleInspectFieldRow("inspectfields", "array", "Recommended top-level compare-bundle field descriptors"));
        rows.add(bundleInspectFieldRow("inspectfieldsCi", "object", "Compact top-level compare-bundle field summary"));
        rows.add(bundleInspectFieldRow("inspectCommands", "array", "Recommended follow-up commands"));
        rows.add(bundleInspectFieldRow("inspectCommandsCi", "object", "Compact recommended command summary"));
        rows.add(bundleInspectFieldRow("inspectFields", "array", "Machine-readable field descriptors"));
        rows.add(bundleInspectFieldRow("inspectFieldsCi", "object", "Compact field discovery summary"));
        return List.copyOf(rows);
    }

    private static String bundleManifestComparisonSummary(Map<String, Object> comparison) {
        return "comparison nextTokensPreviewChanged=" + String.valueOf(comparison.getOrDefault("nextTokensPreviewChanged", false)) + " leftNextTokensPreview=" + String.valueOf(comparison.getOrDefault("leftNextTokensPreview", List.of())) + " rightNextTokensPreview=" + String.valueOf(comparison.getOrDefault("rightNextTokensPreview", List.of()));
    }

    private static Map<String, Object> bundleInspectFieldRow(String name, String type, String description) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", name);
        row.put("type", type);
        row.put("description", description);
        return row;
    }

    private static Map<String, Object> bundleInspectFieldSummary() {
        List<String> names = new ArrayList<>();
        for (Map<String, Object> row : bundleInspectFieldRows()) {
            names.add(String.valueOf(row.getOrDefault("name", "")));
        }
        return Map.of("summary", "short", "fieldCount", Integer.valueOf(names.size()), "fields", names);
    }

    private static List<Map<String, Object>> summarizeBundleHealthFiles(BundleInput bundle, Map<String, String> options) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Map<String, Object> row : bundleFiles(bundle)) {
            Path filePath = Path.of(String.valueOf(row.getOrDefault("path", "")), new String[0]);
            String section = String.valueOf(row.getOrDefault("section", ""));
            String name = String.valueOf(row.getOrDefault("name", ""));
            String format = detectBundleFileFormat(name);
            boolean exists = Files.isRegularFile(filePath, new LinkOption[0]);
            String status = exists ? "ok" : "missing";
            if (matchesBundleHealthFilters(section, name, format, status, options)) {
                Map<String, Object> healthRow = new LinkedHashMap<>();
                healthRow.put("section", section);
                healthRow.put("name", name);
                healthRow.put("format", format);
                healthRow.put("path", filePath.toAbsolutePath().normalize().toString());
                healthRow.put("exists", Boolean.valueOf(exists));
                healthRow.put("status", status);
                rows.add(healthRow);
            }
        }
        rows.sort(compareBundleHealthRows(options.get("sort"), "name"));
        Integer top = parseTop(options.get("top"));
        if (top != null && top.intValue() >= 0 && top.intValue() < rows.size()) {
            rows = new ArrayList<>(rows.subList(0, top.intValue()));
        }
        return List.copyOf(rows);
    }

    private static List<Map<String, Object>> summarizeBundleHealthGrouped(BundleInput bundle, String by, Map<String, String> options) {
        String str;
        Map<String, int[]> counts = new LinkedHashMap<>();
        for (Map<String, Object> row : bundleFiles(bundle)) {
            String section = String.valueOf(row.getOrDefault("section", ""));
            String name = String.valueOf(row.getOrDefault("name", ""));
            String format = detectBundleFileFormat(name);
            Path filePath = Path.of(String.valueOf(row.getOrDefault("path", "")), new String[0]);
            boolean exists = Files.isRegularFile(filePath, new LinkOption[0]);
            String status = exists ? "ok" : "missing";
            if (matchesBundleHealthFilters(section, name, format, status, options)) {
                switch (by) {
                    case "section":
                        str = section;
                        break;
                    case "format":
                        str = format;
                        break;
                    default:
                        throw new IllegalArgumentException("Unsupported bundleHealth grouping '" + by + "'.");
                }
                String key = str;
                counts.putIfAbsent(key, new int[]{0, 0});
                int[] values = counts.get(key);
                values[0] = values[0] + 1;
                if (!exists) {
                    values[1] = values[1] + 1;
                }
            }
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        counts.forEach((key2, values2) -> {
            LinkedHashMap linkedHashMap = new LinkedHashMap();
            linkedHashMap.put(by, key2);
            linkedHashMap.put("count", Integer.valueOf(values2[0]));
            linkedHashMap.put("missingCount", Integer.valueOf(values2[1]));
            linkedHashMap.put("okCount", Integer.valueOf(values2[0] - values2[1]));
            linkedHashMap.put("healthStatus", values2[1] == 0 ? "ok" : "missing-files");
            rows.add(linkedHashMap);
        });
        List<Map<String, Object>> sortedRows = new ArrayList<>(rows);
        sortedRows.sort(compareBundleHealthRows(options.get("sort"), by));
        Integer top = parseTop(options.get("top"));
        if (top != null && top.intValue() >= 0 && top.intValue() < sortedRows.size()) {
            sortedRows = new ArrayList<>(sortedRows.subList(0, top.intValue()));
        }
        return List.copyOf(sortedRows);
    }

    private static boolean matchesBundleHealthFilters(String section, String name, String format, String status, Map<String, String> options) {
        String expectedSection = options.get("section");
        String expectedName = options.get("name");
        String expectedFormat = options.get("format");
        String expectedStatus = options.get("status");
        if (expectedSection != null && !expectedSection.equalsIgnoreCase(section)) {
            return false;
        }
        if (expectedName != null && !expectedName.equalsIgnoreCase(name)) {
            return false;
        }
        if (expectedFormat == null || expectedFormat.equalsIgnoreCase(format)) {
            return expectedStatus == null || expectedStatus.equalsIgnoreCase(status);
        }
        return false;
    }

    private static List<Map<String, Object>> bundleFiles(BundleInput bundle) {
        Object filesValue = bundle.manifest().get("files");
        if (!(filesValue instanceof List)) {
            return List.of();
        }
        List<?> list = (List) filesValue;
        List<Map<String, Object>> files = new ArrayList<>();
        for (Object item : list) {
            if (item instanceof Map) {
                Map<?, ?> map = (Map) item;
                files.add(asMap(map));
            }
        }
        return List.copyOf(files);
    }

    private static List<Map<String, Object>> selectBundleFiles(BundleInput bundle, Map<String, String> options) {
        List<Map<String, Object>> rows = new ArrayList<>();
        String expectedSection = options.get("section");
        String expectedName = options.get("name");
        String expectedFormat = options.get("format");
        for (Map<String, Object> row : bundleFiles(bundle)) {
            String section = String.valueOf(row.getOrDefault("section", ""));
            String name = String.valueOf(row.getOrDefault("name", ""));
            String format = detectBundleFileFormat(name);
            if (expectedSection == null || expectedSection.equalsIgnoreCase(section)) {
                if (expectedName == null || expectedName.equalsIgnoreCase(name)) {
                    if (expectedFormat == null || expectedFormat.equalsIgnoreCase(format)) {
                        Map<String, Object> normalized = new LinkedHashMap<>(row);
                        normalized.put("format", format);
                        rows.add(normalized);
                    }
                }
            }
        }
        rows.sort(compareBundleFileRows(options.get("sort")));
        Integer top = parseTop(options.get("top"));
        if (top != null && top.intValue() >= 0 && top.intValue() < rows.size()) {
            rows = new ArrayList<>(rows.subList(0, top.intValue()));
        }
        return List.copyOf(rows);
    }

    private static List<Map<String, Object>> summarizeBundleFiles(BundleInput bundle, Map<String, String> options) {
        String strDetectBundleFileFormat;
        String by = options.getOrDefault("by", "section");
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Map<String, Object> row : bundleFiles(bundle)) {
            switch (by) {
                case "section":
                    strDetectBundleFileFormat = String.valueOf(row.getOrDefault("section", ""));
                    break;
                case "format":
                    strDetectBundleFileFormat = detectBundleFileFormat(String.valueOf(row.getOrDefault("name", "")));
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported files summary key '" + by + "'.");
            }
            String key = strDetectBundleFileFormat;
            counts.put(key, Integer.valueOf(counts.getOrDefault(key, 0).intValue() + 1));
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        counts.forEach((key2, count) -> {
            LinkedHashMap linkedHashMap = new LinkedHashMap();
            linkedHashMap.put(by, key2);
            linkedHashMap.put("count", count);
            rows.add(linkedHashMap);
        });
        List<Map<String, Object>> sortedRows = new ArrayList<>(rows);
        sortedRows.sort(compareSummaryRows(options.get("sort"), by));
        Integer top = parseTop(options.get("top"));
        if (top != null && top.intValue() >= 0 && top.intValue() < sortedRows.size()) {
            sortedRows = new ArrayList<>(sortedRows.subList(0, top.intValue()));
        }
        return List.copyOf(sortedRows);
    }

    private static String detectBundleFileFormat(String name) {
        int dot = name.lastIndexOf(46);
        if (dot < 0 || dot == name.length() - 1) {
            return "unknown";
        }
        return name.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    private static Map<String, String> parseSelectorOptions(String selector) {
        Map<String, String> options = new LinkedHashMap<>();
        for (String segment : selector.split(":")) {
            if (!segment.isBlank()) {
                int equalsIndex = segment.indexOf(61);
                if (equalsIndex < 0) {
                    options.put(segment, "true");
                } else {
                    options.put(segment.substring(0, equalsIndex), segment.substring(equalsIndex + 1));
                }
            }
        }
        return options;
    }

    private static Comparator<Map<String, Object>> compareSummaryRows(String sort, String by) {
        String key;
        Comparator<Map<String, Object>> comparatorComparing;
        String normalized = (sort == null || sort.isBlank()) ? by : sort;
        boolean descending = normalized.startsWith("-");
        key = descending ? normalized.substring(1) : normalized;
        switch (key) {
            case "count":
                comparatorComparing = Comparator.comparingInt(row -> {
                    return Integer.parseInt(String.valueOf(row.getOrDefault("count", 0)));
                });
                break;
            case "section":
            case "format":
                comparatorComparing = Comparator.comparing(row2 -> {
                    return String.valueOf(row2.getOrDefault(key, ""));
                });
                break;
            default:
                comparatorComparing = Comparator.comparing(row3 -> {
                    return String.valueOf(row3.getOrDefault(by, ""));
                });
                break;
        }
        Comparator<Map<String, Object>> comparator = comparatorComparing;
        return descending ? comparator.reversed() : comparator;
    }

    private static Comparator<Map<String, Object>> compareBundleFileRows(String sort) {
        boolean descending;
        String key;
        String normalized = (sort == null || sort.isBlank()) ? "name" : sort;
        descending = normalized.startsWith("-");
        key = descending ? normalized.substring(1) : normalized;
        switch (key) {
            case "name":
            case "section":
            case "format":
            case "path":
                Comparator<Map<String, Object>> comparator = Comparator.comparing(row -> {
                    return String.valueOf(row.getOrDefault(key, ""));
                });
                return descending ? comparator.reversed() : comparator;
            default:
                throw new IllegalArgumentException("Unsupported files sort field '" + key + "'. Use sort=name, sort=section, sort=format, or sort=path.");
        }
    }

    private static Comparator<Map<String, Object>> compareBundleHealthRows(String sort, String fallback) {
        String key;
        Comparator<Map<String, Object>> comparatorComparing;
        String normalized = (sort == null || sort.isBlank()) ? fallback : sort;
        boolean descending = normalized.startsWith("-");
        key = descending ? normalized.substring(1) : normalized;
        switch (key) {
            case "count":
            case "missingCount":
            case "okCount":
                comparatorComparing = Comparator.comparingInt(row -> {
                    return Integer.parseInt(String.valueOf(row.getOrDefault(key, 0)));
                });
                break;
            case "exists":
                comparatorComparing = Comparator.comparing(row2 -> {
                    return Boolean.valueOf(Boolean.parseBoolean(String.valueOf(row2.getOrDefault("exists", false))));
                });
                break;
            case "section":
            case "format":
            case "name":
            case "status":
            case "healthStatus":
                comparatorComparing = Comparator.comparing(row3 -> {
                    return String.valueOf(row3.getOrDefault(key, ""));
                });
                break;
            default:
                comparatorComparing = Comparator.comparing(row4 -> {
                    return String.valueOf(row4.getOrDefault(fallback, ""));
                });
                break;
        }
        Comparator<Map<String, Object>> comparator = comparatorComparing;
        return descending ? comparator.reversed() : comparator;
    }

    private static Integer parseTop(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Integer.valueOf(Integer.parseInt(value));
    }

    private static Map<String, Object> bundleFileEntry(BundleInput bundle, String name) {
        for (Map<String, Object> row : bundleFiles(bundle)) {
            if (name.equals(String.valueOf(row.getOrDefault("name", "")))) {
                return row;
            }
        }
        throw new IllegalArgumentException("Bundle file not found for name '" + name + "'.");
    }

    private static Object loadBundleFile(BundleInput bundle, String selector) {
        Map<String, Object> row = resolveBundleFile(bundle, selector);
        Path filePath = Path.of(String.valueOf(row.get("path")), new String[0]);
        if (!Files.isRegularFile(filePath, new LinkOption[0])) {
            throw new IllegalArgumentException("Bundle file path not found: " + String.valueOf(filePath.toAbsolutePath().normalize()));
        }
        return loadJsonValue(filePath);
    }

    private static Map<String, Object> resolveBundleFile(BundleInput bundle, String selector) {
        String[] parts = selector.trim().split(":");
        String expectedSection = null;
        String expectedName = null;
        String pick = "first";
        Integer index = null;
        for (String rawPart : parts) {
            String part = rawPart.trim();
            if (!part.isBlank()) {
                String normalized = part.toLowerCase(Locale.ROOT);
                if (normalized.startsWith("section=")) {
                    expectedSection = part.substring("section=".length()).trim();
                } else if (normalized.startsWith("name=")) {
                    expectedName = part.substring("name=".length()).trim();
                } else if (normalized.startsWith("pick=")) {
                    pick = normalized.substring("pick=".length()).trim();
                } else if (normalized.startsWith("index=")) {
                    index = Integer.valueOf(Integer.parseInt(normalized.substring("index=".length()).trim()));
                } else {
                    return resolveBundleFileBySectionOrName(bundle, part, "first", null);
                }
            }
        }
        if (expectedSection != null) {
            return resolveBundleFileBySectionOrName(bundle, expectedSection, pick, index);
        }
        if (expectedName != null) {
            return bundleFileEntry(bundle, expectedName);
        }
        return resolveBundleFileBySectionOrName(bundle, selector.trim(), "first", null);
    }

    private static Map<String, Object> resolveBundleFileBySectionOrName(BundleInput bundle, String sectionOrName, String pick, Integer index) {
        List<Map<String, Object>> sectionMatches = new ArrayList<>();
        for (Map<String, Object> candidate : bundleFiles(bundle)) {
            String section = String.valueOf(candidate.getOrDefault("section", ""));
            if (sectionOrName.equals(section)) {
                sectionMatches.add(candidate);
            }
        }
        if (!sectionMatches.isEmpty()) {
            if (index != null) {
                if (index.intValue() < 0 || index.intValue() >= sectionMatches.size()) {
                    throw new IllegalArgumentException("Bundle section index out of range: " + index);
                }
                return sectionMatches.get(index.intValue());
            }
            if ("last".equalsIgnoreCase(pick)) {
                return sectionMatches.get(sectionMatches.size() - 1);
            }
            if ("first".equalsIgnoreCase(pick)) {
                return sectionMatches.get(0);
            }
            throw new IllegalArgumentException("Unsupported loadfile pick '" + pick + "'. Use pick=first, pick=last, or index=<n>.");
        }
        if (index != null) {
            throw new IllegalArgumentException("Bundle section not found: " + sectionOrName);
        }
        return bundleFileEntry(bundle, sectionOrName);
    }

    private static Map<String, Object> summarizeOverview(InputBundle left, InputBundle right) {
        Map<String, Object> overview = new LinkedHashMap<>();
        overview.put("reportVersion", COMPARE_BUNDLE_REPORT_VERSION);
        overview.put("leftReportFile", left.reportFile().toAbsolutePath().normalize().toString());
        overview.put("rightReportFile", right.reportFile().toAbsolutePath().normalize().toString());
        overview.put("leftReportVersion", left.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        overview.put("rightReportVersion", right.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        overview.put("leftPrompt", left.report().getOrDefault("prompt", ""));
        overview.put("rightPrompt", right.report().getOrDefault("prompt", ""));
        overview.put("leftEpochCount", selectObjectPath(left.report().get("training"), "epochCount"));
        overview.put("rightEpochCount", selectObjectPath(right.report().get("training"), "epochCount"));
        overview.put("leftLatestTrainLoss", selectObjectPath(left.report().get("training"), "latestTrainLoss"));
        overview.put("rightLatestTrainLoss", selectObjectPath(right.report().get("training"), "latestTrainLoss"));
        overview.put("leftNextToken", selectObjectPath(left.report().get("inference"), "nextToken"));
        overview.put("rightNextToken", selectObjectPath(right.report().get("inference"), "nextToken"));
        overview.put("leftNextTokens", selectObjectPath(left.report().get("inference"), "nextTokens"));
        overview.put("rightNextTokens", selectObjectPath(right.report().get("inference"), "nextTokens"));
        overview.put("leftNextTokensPreview", readNextTokensPreview(asMap(left.report().get("inference"))));
        overview.put("rightNextTokensPreview", readNextTokensPreview(asMap(right.report().get("inference"))));
        overview.put("leftNextTokensCount", Integer.valueOf(sizeOfValue(selectObjectPath(left.report().get("inference"), "nextTokens"))));
        overview.put("rightNextTokensCount", Integer.valueOf(sizeOfValue(selectObjectPath(right.report().get("inference"), "nextTokens"))));
        overview.put("leftCombinedText", selectObjectPath(left.report().get("inference"), "combinedText"));
        overview.put("rightCombinedText", selectObjectPath(right.report().get("inference"), "combinedText"));
        overview.put("latestTrainLossDelta", Double.valueOf(difference(asDouble(selectObjectPath(left.report().get("training"), "latestTrainLoss")), asDouble(selectObjectPath(right.report().get("training"), "latestTrainLoss")))));
        overview.put("nextTokenChanged", Boolean.valueOf(!String.valueOf(selectObjectPath(left.report().get("inference"), "nextToken")).equals(String.valueOf(selectObjectPath(right.report().get("inference"), "nextToken")))));
        overview.put("nextTokensChanged", Boolean.valueOf(!String.valueOf(selectObjectPath(left.report().get("inference"), "nextTokens")).equals(String.valueOf(selectObjectPath(right.report().get("inference"), "nextTokens")))));
        overview.put("nextTokensPreviewChanged", Boolean.valueOf(!String.valueOf(readNextTokensPreview(asMap(left.report().get("inference")))).equals(String.valueOf(readNextTokensPreview(asMap(right.report().get("inference")))))));
        overview.put("nextTokensCountDelta", Double.valueOf(difference(sizeOfValue(selectObjectPath(left.report().get("inference"), "nextTokens")), sizeOfValue(selectObjectPath(right.report().get("inference"), "nextTokens")))));
        overview.put("combinedTextChanged", Boolean.valueOf(!String.valueOf(selectObjectPath(left.report().get("inference"), "combinedText")).equals(String.valueOf(selectObjectPath(right.report().get("inference"), "combinedText")))));
        return overview;
    }

    private static Map<String, Object> summarizeStatus(InputBundle left, InputBundle right, boolean ci) {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("summary", ci ? "short" : "full");
        status.put("reportVersion", COMPARE_BUNDLE_REPORT_VERSION);
        status.put("leftReportFile", left.reportFile().toAbsolutePath().normalize().toString());
        status.put("rightReportFile", right.reportFile().toAbsolutePath().normalize().toString());
        status.put("leftReportVersion", left.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        status.put("rightReportVersion", right.report().getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        status.put("leftLatestTrainLoss", selectObjectPath(left.report().get("training"), "latestTrainLoss"));
        status.put("rightLatestTrainLoss", selectObjectPath(right.report().get("training"), "latestTrainLoss"));
        status.put("latestTrainLossDelta", Double.valueOf(difference(asDouble(selectObjectPath(left.report().get("training"), "latestTrainLoss")), asDouble(selectObjectPath(right.report().get("training"), "latestTrainLoss")))));
        status.put("leftNextToken", selectObjectPath(left.report().get("inference"), "nextToken"));
        status.put("rightNextToken", selectObjectPath(right.report().get("inference"), "nextToken"));
        status.put("leftNextTokensPreview", readNextTokensPreview(asMap(left.report().get("inference"))));
        status.put("rightNextTokensPreview", readNextTokensPreview(asMap(right.report().get("inference"))));
        status.put("leftNextTokensCount", Integer.valueOf(sizeOfValue(selectObjectPath(left.report().get("inference"), "nextTokens"))));
        status.put("rightNextTokensCount", Integer.valueOf(sizeOfValue(selectObjectPath(right.report().get("inference"), "nextTokens"))));
        status.put("nextTokensChanged", Boolean.valueOf(!String.valueOf(selectObjectPath(left.report().get("inference"), "nextTokens")).equals(String.valueOf(selectObjectPath(right.report().get("inference"), "nextTokens")))));
        status.put("nextTokensPreviewChanged", Boolean.valueOf(!String.valueOf(readNextTokensPreview(asMap(left.report().get("inference")))).equals(String.valueOf(readNextTokensPreview(asMap(right.report().get("inference")))))));
        status.put("nextTokensCountDelta", Double.valueOf(difference(sizeOfValue(selectObjectPath(left.report().get("inference"), "nextTokens")), sizeOfValue(selectObjectPath(right.report().get("inference"), "nextTokens")))));
        status.put("combinedTextChanged", Boolean.valueOf(!String.valueOf(selectObjectPath(left.report().get("inference"), "combinedText")).equals(String.valueOf(selectObjectPath(right.report().get("inference"), "combinedText")))));
        return status;
    }

    private static Map<String, Object> comparisonBlock(InputBundle left, InputBundle right) {
        return summarizeStatus(left, right, false);
    }

    private static String compareSummaryLine(InputBundle left, InputBundle right) {
        Map<String, Object> status = compareSummary(left, right);
        return "compareSummary reportVersion=" + String.valueOf(status.getOrDefault("reportVersion", COMPARE_BUNDLE_REPORT_VERSION)) + " leftReportVersion=" + String.valueOf(status.getOrDefault("leftReportVersion", SINGLE_REPORT_VERSION_FALLBACK)) + " rightReportVersion=" + String.valueOf(status.getOrDefault("rightReportVersion", SINGLE_REPORT_VERSION_FALLBACK)) + " latestTrainLossDelta=" + String.valueOf(status.getOrDefault("latestTrainLossDelta", "")) + " nextTokensPreviewChanged=" + String.valueOf(status.getOrDefault("nextTokensPreviewChanged", false)) + " nextTokensCountDelta=" + String.valueOf(status.getOrDefault("nextTokensCountDelta", "")) + " combinedTextChanged=" + String.valueOf(status.getOrDefault("combinedTextChanged", false));
    }

    private static Map<String, Object> compareArtifactSummary(InputBundle left, InputBundle right) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "short");
        summary.put("reportVersion", COMPARE_BUNDLE_REPORT_VERSION);
        summary.put("leftReportFile", left.reportFile().toAbsolutePath().normalize().toString());
        summary.put("rightReportFile", right.reportFile().toAbsolutePath().normalize().toString());
        summary.put("compareSummary", compareSummary(left, right));
        summary.put("compareSummaryLine", compareSummaryLine(left, right));
        summary.put("commandsCi", compareCommands(left, right, true));
        summary.put("selectorsCi", compareSelectors(left, right, true));
        summary.put("inspectfieldsCi", compareInspectFields(true));
        return summary;
    }

    private static String compareArtifactSummaryLine(InputBundle left, InputBundle right) {
        Map<String, Object> summary = compareArtifactSummary(left, right);
        return "artifactSummary reportVersion=" + String.valueOf(summary.getOrDefault("reportVersion", COMPARE_BUNDLE_REPORT_VERSION)) + " leftReportFile=" + String.valueOf(summary.getOrDefault("leftReportFile", "")) + " rightReportFile=" + String.valueOf(summary.getOrDefault("rightReportFile", "")) + " compareSummaryLine=" + String.valueOf(summary.getOrDefault("compareSummaryLine", ""));
    }

    private static Object compareCommands(InputBundle left, InputBundle right, boolean ci) {
        String leftTarget = left.inputPath().toAbsolutePath().normalize().toString();
        String rightTarget = right.inputPath().toAbsolutePath().normalize().toString();
        Map<String, Object> commands = new LinkedHashMap<>();
        commands.put("overview", compareCommandText(leftTarget, rightTarget, "overview"));
        commands.put("artifactSummary", compareCommandText(leftTarget, rightTarget, "artifactSummary"));
        commands.put("artifactSummaryLine", compareCommandText(leftTarget, rightTarget, "artifactSummaryLine"));
        commands.put("comparison", compareCommandText(leftTarget, rightTarget, "comparison"));
        commands.put("compareSummary", compareCommandText(leftTarget, rightTarget, "compareSummary"));
        commands.put("compareSummaryLine", compareCommandText(leftTarget, rightTarget, "compareSummaryLine"));
        commands.put("status", compareCommandText(leftTarget, rightTarget, "status"));
        commands.put("nextTokensPreview", compareCommandText(leftTarget, rightTarget, "delta:nextTokensPreview"));
        commands.put("combinedText", compareCommandText(leftTarget, rightTarget, "delta:combinedText"));
        if (!ci) {
            return commands;
        }
        return Map.of("summary", "short", "commandCount", Integer.valueOf(commands.size()), "commands", new ArrayList(commands.values()));
    }

    private static Object compareSelectors(InputBundle left, InputBundle right, boolean ci) {
        String leftTarget = left.inputPath().toAbsolutePath().normalize().toString();
        String rightTarget = right.inputPath().toAbsolutePath().normalize().toString();
        Map<String, Object> selectors = new LinkedHashMap<>();
        selectors.put("commands", compareCommandText(leftTarget, rightTarget, "commands:ci"));
        selectors.put("artifactSummary", compareCommandText(leftTarget, rightTarget, "artifactSummaryLine"));
        selectors.put("comparison", compareCommandText(leftTarget, rightTarget, "comparison"));
        selectors.put("compareSummary", compareCommandText(leftTarget, rightTarget, "compareSummary"));
        selectors.put("compareSummaryLine", compareCommandText(leftTarget, rightTarget, "compareSummaryLine"));
        selectors.put("overview", compareCommandText(leftTarget, rightTarget, "overview"));
        selectors.put("status", compareCommandText(leftTarget, rightTarget, "status"));
        selectors.put("deltaNextTokensPreview", compareCommandText(leftTarget, rightTarget, "delta:nextTokensPreview"));
        if (!ci) {
            return selectors;
        }
        return Map.of("summary", "short", "selectorCount", Integer.valueOf(selectors.size()), "selectors", new ArrayList(selectors.values()));
    }

    private static Object compareInspectFields(boolean ci) {
        if (!ci) {
            return compareInspectFieldRows();
        }
        return compareInspectFieldSummary();
    }

    private static List<Map<String, Object>> compareInspectFieldRows() {
        List<Map<String, Object>> rows = new ArrayList<>();
        rows.add(compareInspectFieldRow("reportVersion", "string", "Compare output schema marker"));
        rows.add(compareInspectFieldRow("leftReportVersion", "string", "Left train-infer report schema marker"));
        rows.add(compareInspectFieldRow("rightReportVersion", "string", "Right train-infer report schema marker"));
        rows.add(compareInspectFieldRow("leftReportFile", "string", "Left train-infer report path"));
        rows.add(compareInspectFieldRow("rightReportFile", "string", "Right train-infer report path"));
        rows.add(compareInspectFieldRow("overview", "object", "Live compare overview snapshot"));
        rows.add(compareInspectFieldRow("comparison", "object", "Shared compact comparison block"));
        rows.add(compareInspectFieldRow("artifactSummary", "object", "Compact live compare artifact dashboard"));
        rows.add(compareInspectFieldRow("artifactSummaryLine", "string", "Human-readable one-line live compare artifact dashboard"));
        rows.add(compareInspectFieldRow("compareSummary", "object", "Compact machine-readable compare summary"));
        rows.add(compareInspectFieldRow("compareSummaryLine", "string", "Human-readable one-line compare summary"));
        rows.add(compareInspectFieldRow("commands", "object", "Recommended live compare commands"));
        rows.add(compareInspectFieldRow("commandsCi", "object", "Compact live compare command summary"));
        rows.add(compareInspectFieldRow("status", "object", "Compact compare status block"));
        rows.add(compareInspectFieldRow("runs", "array", "Per-side run summaries"));
        rows.add(compareInspectFieldRow("selectors", "object", "Recommended live compare selectors"));
        rows.add(compareInspectFieldRow("selectorsCi", "object", "Compact live compare selector summary"));
        rows.add(compareInspectFieldRow("delta:latestTrainLoss", "object", "Delta view for latest training loss"));
        rows.add(compareInspectFieldRow("delta:bestValidationLoss", "object", "Delta view for best validation loss"));
        rows.add(compareInspectFieldRow("delta:epochCount", "object", "Delta view for epoch count"));
        rows.add(compareInspectFieldRow("delta:nextToken", "object", "Delta view for next-token prediction"));
        rows.add(compareInspectFieldRow("delta:nextTokens", "object", "Delta view for ranked next-token candidates"));
        rows.add(compareInspectFieldRow("delta:nextTokensPreview", "object", "Delta view for compact next-token preview"));
        rows.add(compareInspectFieldRow("delta:nextTokensCount", "object", "Delta view for next-token candidate count"));
        rows.add(compareInspectFieldRow("delta:generatedText", "object", "Delta view for generated text"));
        rows.add(compareInspectFieldRow("delta:combinedText", "object", "Delta view for combined prompt-plus-generated text"));
        return List.copyOf(rows);
    }

    private static Map<String, Object> compareInspectFieldRow(String name, String type, String description) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", name);
        row.put("type", type);
        row.put("description", description);
        return row;
    }

    private static Map<String, Object> compareInspectFieldSummary() {
        List<String> names = new ArrayList<>();
        for (Map<String, Object> row : compareInspectFieldRows()) {
            names.add(String.valueOf(row.getOrDefault("name", "")));
        }
        return Map.of("summary", "short", "fieldCount", Integer.valueOf(names.size()), "fields", names);
    }

    private static Map<String, Object> compareSummary(InputBundle left, InputBundle right) {
        Map<String, Object> status = summarizeStatus(left, right, true);
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("summary", "short");
        summary.put("reportVersion", status.getOrDefault("reportVersion", COMPARE_BUNDLE_REPORT_VERSION));
        summary.put("leftReportVersion", status.getOrDefault("leftReportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        summary.put("rightReportVersion", status.getOrDefault("rightReportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        summary.put("leftReportFile", status.getOrDefault("leftReportFile", ""));
        summary.put("rightReportFile", status.getOrDefault("rightReportFile", ""));
        summary.put("latestTrainLossDelta", status.getOrDefault("latestTrainLossDelta", ""));
        summary.put("nextTokensPreviewChanged", status.getOrDefault("nextTokensPreviewChanged", false));
        summary.put("nextTokensCountDelta", status.getOrDefault("nextTokensCountDelta", ""));
        summary.put("combinedTextChanged", status.getOrDefault("combinedTextChanged", false));
        return summary;
    }

    private static Map<String, Object> summarizeRun(String label, InputBundle input) {
        Map<String, Object> report = input.report();
        Map<String, Object> training = asMap(report.get("training"));
        Map<String, Object> inference = asMap(report.get("inference"));
        Map<String, Object> metadata = asMap(training.get("metadata"));
        Map<String, Object> run = new LinkedHashMap<>();
        run.put("reportVersion", report.getOrDefault("reportVersion", SINGLE_REPORT_VERSION_FALLBACK));
        run.put("label", label);
        run.put("reportFile", input.reportFile().toAbsolutePath().normalize().toString());
        run.put("prompt", report.getOrDefault("prompt", ""));
        run.put("epochCount", training.getOrDefault("epochCount", ""));
        run.put("latestTrainLoss", training.getOrDefault("latestTrainLoss", ""));
        run.put("bestValidationLoss", training.getOrDefault("bestValidationLoss", ""));
        run.put("globalStep", metadata.getOrDefault("globalStep", ""));
        run.put("nextToken", inference.getOrDefault("nextToken", ""));
        run.put("nextTokens", inference.getOrDefault("nextTokens", List.of()));
        run.put("nextTokensPreview", readNextTokensPreview(inference));
        run.put("nextTokensCount", Integer.valueOf(sizeOfValue(inference.get("nextTokens"))));
        run.put("generatedText", inference.getOrDefault("generatedText", ""));
        run.put("combinedText", inference.getOrDefault("combinedText", ""));
        return run;
    }

    private static Map<String, Object> summarizeNumericDelta(String field, double left, double right) {
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("field", field);
        delta.put("left", Double.valueOf(left));
        delta.put("right", Double.valueOf(right));
        delta.put("delta", Double.valueOf(difference(left, right)));
        delta.put("changed", Boolean.valueOf(Double.compare(left, right) != 0));
        return delta;
    }

    private static Map<String, Object> summarizeValueDelta(String field, Object left, Object right) {
        Map<String, Object> delta = new LinkedHashMap<>();
        delta.put("field", field);
        delta.put("left", left);
        delta.put("right", right);
        delta.put("changed", Boolean.valueOf(!String.valueOf(left).equals(String.valueOf(right))));
        return delta;
    }

    private static int sizeOfValue(Object value) {
        if (value instanceof List) {
            List<?> list = (List) value;
            return list.size();
        }
        if (value instanceof Map) {
            Map<?, ?> map = (Map) value;
            return map.size();
        }
        if (value == null) {
            return 0;
        }
        return 1;
    }

    private static Object readNextTokensPreview(Map<String, Object> inference) {
        Object preview = inference.get("nextTokensPreview");
        if (preview instanceof List) {
            List<?> list = (List) preview;
            if (!list.isEmpty()) {
                return preview;
            }
        }
        return nextTokensPreview(inference.get("nextTokens"), 3);
    }

    private static List<String> nextTokensPreview(Object value, int limit) {
        if (value instanceof List) {
            List<?> list = (List) value;
            if (!list.isEmpty() && limit > 0) {
                List<String> preview = new ArrayList<>();
                int size = Math.min(limit, list.size());
                for (int index = 0; index < size; index++) {
                    Object item = list.get(index);
                    if (item instanceof Map) {
                        Map<?, ?> map = (Map) item;
                        Object rank = map.get("rank");
                        Object tokenId = map.get("tokenId");
                        Object score = map.get("score");
                        String tokenText = tokenId == null ? "?" : String.valueOf(tokenId);
                        String rankText = rank == null ? String.valueOf(index + 1) : String.valueOf(rank);
                        String scoreText = score == null ? "" : "@" + formatScore(score);
                        preview.add(rankText + ":" + tokenText + scoreText);
                    } else {
                        preview.add(String.valueOf(item));
                    }
                }
                return preview;
            }
        }
        return List.of();
    }

    private static String formatScore(Object value) {
        if (!(value instanceof Number)) {
            return String.valueOf(value);
        }
        Number number = (Number) value;
        return String.format(Locale.ROOT, "%.4f", Double.valueOf(number.doubleValue()));
    }

    private static Map<String, Object> loadJsonMap(Path file) {
        try {
            Map<String, Object> values = (Map) JSON.readValue(file.toFile(), LinkedHashMap.class);
            return values;
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to load JSON file " + String.valueOf(file) + ".", exception);
        }
    }

    private static Object loadJsonValue(Path file) {
        try {
            return JSON.readValue(file.toFile(), Object.class);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to load JSON file " + String.valueOf(file) + ".", exception);
        }
    }

    private static Object selectObjectPath(Object root, String path) {
        Object current = root;
        for (String segment : path.split(":")) {
            if (!segment.isBlank()) {
                if (current instanceof Map) {
                    Map<?, ?> map = (Map) current;
                    current = map.get(segment);
                } else {
                    throw new IllegalArgumentException("Cannot descend into path segment '" + segment + "'.");
                }
            }
        }
        return current;
    }

    private static Map<String, Object> asMap(Object value) {
        if (!(value instanceof Map)) {
            return Map.of();
        }
        Map<?, ?> map = (Map) value;
        return asMap(map);
    }

    private static Map<String, Object> asMap(Map<?, ?> map) {
        Map<String, Object> copy = new LinkedHashMap<>();
        map.forEach((key, element) -> {
            copy.put(String.valueOf(key), element);
        });
        return copy;
    }

    private static double asDouble(Object value) {
        if (value instanceof Number) {
            Number number = (Number) value;
            return number.doubleValue();
        }
        return Double.parseDouble(String.valueOf(value));
    }

    private static int numberAsInt(Object value) {
        if (value instanceof Number) {
            Number number = (Number) value;
            return number.intValue();
        }
        return Integer.parseInt(String.valueOf(value));
    }

    private static double difference(double left, double right) {
        return right - left;
    }

    private static String normalizeFormat(String format, Path outputPath) {
        if (format != null && !format.isBlank()) {
            return format.toLowerCase(Locale.ROOT);
        }
        if (outputPath != null) {
            String name = outputPath.getFileName().toString().toLowerCase(Locale.ROOT);
            if (name.endsWith(".json")) {
                return "json";
            }
            if (name.endsWith(".csv")) {
                return "csv";
            }
            return "text";
        }
        return "text";
    }

    private static String renderValue(Object value, String format) {
        switch (format) {
            case "json":
                return renderJson(value);
            case "csv":
                return renderCsv(value);
            case "text":
                return renderText(value);
            default:
                throw new IllegalArgumentException("Unsupported format '" + format + "'. Use text, json, or csv.");
        }
    }

    private static String renderText(Object value) {
        if (value instanceof Map) {
            Map<?, ?> map = (Map) value;
            if (isStatusMap(map)) {
                return renderStatusText(asMap(map));
            }
            StringBuilder out = new StringBuilder();
            map.forEach((key, element) -> {
                out.append(key).append('=').append(String.valueOf(element)).append('\n');
            });
            return out.toString();
        }
        if (value instanceof List) {
            List<?> list = (List) value;
            if (list.isEmpty()) {
                return "(empty)\n";
            }
            if (list.get(0) instanceof Map) {
                return renderTextTable(asMapList(list));
            }
            StringBuilder out2 = new StringBuilder();
            for (Object item : list) {
                out2.append(String.valueOf(item)).append('\n');
            }
            return out2.toString();
        }
        return String.valueOf(value) + "\n";
    }

    private static boolean isStatusMap(Map<?, ?> map) {
        return map.containsKey("summary") && ((map.containsKey("leftReportFile") && map.containsKey("rightReportFile")) || (map.containsKey("bundleDir") && map.containsKey("healthStatus")));
    }

    private static String renderStatusText(Map<String, Object> map) {
        String summary = String.valueOf(map.getOrDefault("summary", "full"));
        if ("short".equals(summary)) {
            if (map.containsKey("bundleDir") && map.containsKey("healthStatus")) {
                return "status healthStatus=" + String.valueOf(map.getOrDefault("healthStatus", "")) + " fileCount=" + String.valueOf(map.getOrDefault("fileCount", 0)) + " missingFileCount=" + String.valueOf(map.getOrDefault("missingFileCount", 0)) + " bundleDir=" + String.valueOf(map.getOrDefault("bundleDir", "")) + "\n";
            }
            return "status leftLatestTrainLoss=" + String.valueOf(map.getOrDefault("leftLatestTrainLoss", "")) + " rightLatestTrainLoss=" + String.valueOf(map.getOrDefault("rightLatestTrainLoss", "")) + " latestTrainLossDelta=" + String.valueOf(map.getOrDefault("latestTrainLossDelta", "")) + " leftNextToken=" + String.valueOf(map.getOrDefault("leftNextToken", "")) + " rightNextToken=" + String.valueOf(map.getOrDefault("rightNextToken", "")) + " combinedTextChanged=" + String.valueOf(map.getOrDefault("combinedTextChanged", "")) + "\n";
        }
        StringBuilder out = new StringBuilder();
        map.forEach((key, value) -> {
            out.append(key).append('=').append(String.valueOf(value)).append('\n');
        });
        return out.toString();
    }

    private static String renderJson(Object value) {
        try {
            return JSON.writeValueAsString(value) + "\n";
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to render JSON output.", exception);
        }
    }

    private static String renderCsv(Object value) {
        if (value instanceof Map) {
            Map<?, ?> map = (Map) value;
            return renderCsvRows(List.of(asMap(map)));
        }
        if (value instanceof List) {
            List<?> list = (List) value;
            if (list.isEmpty()) {
                return "";
            }
            if (!(list.get(0) instanceof Map)) {
                throw new IllegalArgumentException("CSV output requires a map or list of maps.");
            }
            return renderCsvRows(asMapList(list));
        }
        throw new IllegalArgumentException("CSV output requires a map or list of maps.");
    }

    private static List<Map<String, Object>> asMapList(List<?> rows) {
        List<Map<String, Object>> maps = new ArrayList<>();
        for (Object row : rows) {
            if (!(row instanceof Map)) {
                throw new IllegalArgumentException("Expected a list of maps.");
            }
            maps.add(asMap((Map<?, ?>) row));
        }
        return maps;
    }

    private static String renderTextTable(List<Map<String, Object>> rows) {
        List<String> headers = collectHeaders(rows);
        List<Integer> widths = new ArrayList<>();
        for (String header : headers) {
            int width = header.length();
            for (Map<String, Object> row : rows) {
                width = Math.max(width, String.valueOf(row.getOrDefault(header, "")).length());
            }
            widths.add(Integer.valueOf(width));
        }
        StringBuilder out = new StringBuilder();
        appendTableRow(out, headers, widths);
        appendSeparator(out, widths);
        for (Map<String, Object> row2 : rows) {
            List<String> cells = headers.stream().map(header2 -> {
                return String.valueOf(row2.getOrDefault(header2, ""));
            }).toList();
            appendTableRow(out, cells, widths);
        }
        return out.toString();
    }

    private static String renderCsvRows(List<Map<String, Object>> rows) {
        List<String> headers = collectHeaders(rows);
        StringBuilder out = new StringBuilder();
        out.append(String.join(",", headers.stream().map(trainer_byte_latent_train_infer_compare_inspector::escapeCsv).toList())).append('\n');
        for (Map<String, Object> row : rows) {
            out.append(String.join(",", headers.stream().map(header -> {
                return escapeCsv(String.valueOf(row.getOrDefault(header, "")));
            }).toList())).append('\n');
        }
        return out.toString();
    }

    private static List<String> collectHeaders(List<Map<String, Object>> rows) {
        LinkedHashSet<String> headers = new LinkedHashSet<>();
        for (Map<String, Object> row : rows) {
            headers.addAll(row.keySet());
        }
        return List.copyOf(headers);
    }

    private static void appendTableRow(StringBuilder out, List<String> cells, List<Integer> widths) {
        for (int index = 0; index < cells.size(); index++) {
            if (index > 0) {
                out.append(" | ");
            }
            out.append(padRight(cells.get(index), widths.get(index).intValue()));
        }
        out.append('\n');
    }

    private static void appendSeparator(StringBuilder out, List<Integer> widths) {
        for (int index = 0; index < widths.size(); index++) {
            if (index > 0) {
                out.append("-+-");
            }
            out.append("-".repeat(widths.get(index).intValue()));
        }
        out.append('\n');
    }

    private static String padRight(String text, int width) {
        return text + " ".repeat(Math.max(0, width - text.length()));
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static String escapeCsv(String value) {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    public static String escapeTsv(String value) {
        return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n");
    }

    public static String escapeMarkdownTableCell(String value) {
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ");
    }

    public static String escapeMarkdownTableCode(String value) {
        return value.replace("`", "\\`").replace("\r", " ").replace("\n", " ");
    }

    public static String sha256Hex(byte[] bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashed = digest.digest(bytes);
            StringBuilder hex = new StringBuilder(hashed.length * 2);
            for (byte value : hashed) {
                hex.append(String.format("%02x", value & 255));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is not available.", exception);
        }
    }

    private static void writeOutput(Path outputPath, String rendered) {
        try {
            Path parent = outputPath.toAbsolutePath().normalize().getParent();
            if (parent != null) {
                Files.createDirectories(parent, new FileAttribute[0]);
            }
            Files.writeString(outputPath, rendered, StandardCharsets.UTF_8, new OpenOption[0]);
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to write output to " + String.valueOf(outputPath) + ".", exception);
        }
    }

    private static void emitValue(Object value, String format, Path outputPath) {
        String rendered = renderValue(value, format);
        if (outputPath != null) {
            writeOutput(outputPath, rendered);
            System.out.println("wroteOutput=" + String.valueOf(outputPath.toAbsolutePath().normalize()));
        } else {
            System.out.print(rendered);
        }
    }

    private static CliOptions parseArgs(String[] args, boolean bundleMode) {
        String str;
        String str2;
        Path pathOf;
        if (args.length >= (bundleMode ? 2 : 3)) {
            str = args[bundleMode ? (char) 1 : (char) 2];
        } else {
            str = bundleMode ? "bundleSummary" : "overview";
        }
        String section = str;
        if (args.length >= (bundleMode ? 3 : 4)) {
            str2 = args[bundleMode ? (char) 2 : (char) 3];
        } else {
            str2 = null;
        }
        String format = str2;
        if (args.length >= (bundleMode ? 4 : 5)) {
            pathOf = Path.of(args[bundleMode ? (char) 3 : (char) 4], new String[0]);
        } else {
            pathOf = null;
        }
        Path outputPath = pathOf;
        return new CliOptions(section, format, outputPath);
    }

    private record CliOptions(String section, String format, Path outputPath) {
    }

    private record InputBundle(Path inputPath, Path reportFile, Map<String, Object> report) {
    }

    private record BundleInput(Path bundleDir, Path manifestFile, Map<String, Object> manifest) {
    }

    private record ResultInput(Path resultFile, Map<String, Object> result) {
    }
}
