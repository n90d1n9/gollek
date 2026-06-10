package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * One-call writer/reader/verifier for complete portfolio export artifact packages.
 */
public final class TrainingReportPortfolioArtifactPackage {
    private static final String JSON_ARTIFACT = "json";
    private static final String MARKDOWN_ARTIFACT = "markdown";
    private static final String LEADERBOARD_CSV_ARTIFACT = "leaderboardCsv";
    private static final String COMPARISON_METRICS_CSV_ARTIFACT = "comparisonMetricsCsv";
    private static final String COMPARISON_FINDINGS_CSV_ARTIFACT = "comparisonFindingsCsv";

    private TrainingReportPortfolioArtifactPackage() {
    }

    public record Options(
            TrainingReportPortfolioArtifacts.Options artifacts,
            TrainingReportPortfolioArtifactManifest.Options manifest) {
        public Options {
            artifacts = artifacts == null ? TrainingReportPortfolioArtifacts.Options.defaults() : artifacts;
            manifest = manifest == null ? TrainingReportPortfolioArtifactManifest.Options.defaults() : manifest;
        }

        public static Options defaults() {
            return new Options(
                    TrainingReportPortfolioArtifacts.Options.defaults(),
                    TrainingReportPortfolioArtifactManifest.Options.defaults());
        }
    }

    public record PackageBundle(
            Path directory,
            TrainingReportPortfolioArtifacts.ArtifactBundle artifacts,
            TrainingReportPortfolioArtifactManifest.ManifestBundle manifest) {
        public PackageBundle {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
            manifest = Objects.requireNonNull(manifest, "manifest must not be null");
        }

        public int entryCount() {
            return artifacts.export().entryCount();
        }

        public int comparisonMetricCount() {
            return artifacts.export().comparisonMetricCount();
        }

        public int comparisonFindingCount() {
            return artifacts.export().comparisonFindingCount();
        }

        public boolean hasComparisons() {
            return artifacts.hasComparisons();
        }

        public boolean hasComparisonFindings() {
            return artifacts.hasComparisonFindings();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("entryCount", entryCount());
            map.put("comparisonMetricCount", comparisonMetricCount());
            map.put("comparisonFindingCount", comparisonFindingCount());
            map.put("hasComparisons", hasComparisons());
            map.put("hasComparisonFindings", hasComparisonFindings());
            map.put("artifacts", artifacts.toMap());
            map.put("manifest", manifest.toMap());
            return Map.copyOf(map);
        }
    }

    public record PackageInspection(
            Path directory,
            TrainingReportPortfolioArtifactManifest.ManifestInspection manifest,
            TrainingReportPortfolioArtifacts.ArtifactInspection artifacts) {
        public PackageInspection {
            directory = Objects.requireNonNull(directory, "directory must not be null").toAbsolutePath().normalize();
            manifest = Objects.requireNonNull(manifest, "manifest must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
        }

        public int entryCount() {
            return manifest.entryCount();
        }

        public int comparisonMetricCount() {
            return manifest.comparisonMetricCount();
        }

        public int comparisonFindingCount() {
            return manifest.comparisonFindingCount();
        }

        public boolean hasComparisons() {
            return manifest.hasComparisons();
        }

        public boolean hasComparisonFindings() {
            return manifest.hasComparisonFindings();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("entryCount", entryCount());
            map.put("comparisonMetricCount", comparisonMetricCount());
            map.put("comparisonFindingCount", comparisonFindingCount());
            map.put("hasComparisons", hasComparisons());
            map.put("hasComparisonFindings", hasComparisonFindings());
            map.put("manifest", manifest.toMap());
            map.put("artifacts", artifacts.toMap());
            return Map.copyOf(map);
        }
    }

    public record VerifiedPackageBundle(
            PackageBundle bundle,
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification) {
        public VerifiedPackageBundle {
            bundle = Objects.requireNonNull(bundle, "bundle must not be null");
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("passed", passed());
            map.put("bundle", bundle.toMap());
            map.put("verification", verification.toMap());
            return Map.copyOf(map);
        }
    }

    public record PackageRefresh(
            Path directory,
            TrainingReportPortfolioArtifactManifest.ManifestInspection manifest,
            TrainingReportPortfolioArtifacts.ArtifactInspection artifacts,
            TrainingReportPortfolioArtifactManifest.ManifestVerification verification,
            TrainingReportPortfolioArtifactPackageReport.ReportBundle reports,
            TrainingReportPortfolioArtifactPackageReport.ReportVerification reportVerification) {
        public PackageRefresh {
            directory = Objects.requireNonNull(directory, "directory must not be null")
                    .toAbsolutePath()
                    .normalize();
            manifest = Objects.requireNonNull(manifest, "manifest must not be null");
            artifacts = Objects.requireNonNull(artifacts, "artifacts must not be null");
            verification = Objects.requireNonNull(verification, "verification must not be null");
            reports = Objects.requireNonNull(reports, "reports must not be null");
            reportVerification = Objects.requireNonNull(reportVerification, "reportVerification must not be null");
        }

        public boolean passed() {
            return verification.passed() && reports.passed() && reportVerification.passed();
        }

        public void requirePassed() {
            verification.requirePassed();
            reportVerification.requirePassed();
        }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("directory", directory.toString());
            map.put("passed", passed());
            map.put("manifest", manifest.toMap());
            map.put("artifacts", artifacts.toMap());
            map.put("verification", verification.toMap());
            map.put("reports", reports.toMap());
            map.put("reportVerification", reportVerification.toMap());
            return Map.copyOf(map);
        }
    }

    public static PackageBundle write(
            Path directory,
            Map<String, Path> reportFiles) throws IOException {
        return write(directory, TrainingReportPortfolio.fromFiles(reportFiles), Options.defaults(), Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            Map<String, Path> reportFiles,
            String baselineName) throws IOException {
        return write(
                directory,
                TrainingReportPortfolio.fromFiles(reportFiles).exportAgainst(baselineName),
                Options.defaults(),
                Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPortfolio portfolio) throws IOException {
        return write(directory, portfolio, Options.defaults(), Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPortfolio portfolio,
            String baselineName) throws IOException {
        return write(directory, portfolio.exportAgainst(baselineName), Options.defaults(), Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPortfolioExport export) throws IOException {
        return write(directory, export, Options.defaults(), Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPortfolioExport export,
            Options options) throws IOException {
        return write(directory, export, options, Instant.now());
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPortfolioExport export,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");
        TrainingReportPortfolioArtifacts.ArtifactBundle artifacts = TrainingReportPortfolioArtifacts.write(
                resolvedDirectory,
                Objects.requireNonNull(export, "export must not be null"),
                resolvedOptions.artifacts());
        TrainingReportPortfolioArtifactManifest.ManifestBundle manifest =
                TrainingReportPortfolioArtifactManifest.write(
                        artifacts,
                        resolvedOptions.manifest(),
                        resolvedGeneratedAt);
        return new PackageBundle(resolvedDirectory, artifacts, manifest);
    }

    public static PackageBundle write(
            Path directory,
            TrainingReportPortfolio portfolio,
            Options options,
            Instant generatedAt) throws IOException {
        Objects.requireNonNull(portfolio, "portfolio must not be null");
        return write(directory, portfolio.export(), options, generatedAt);
    }

    public static PackageInspection read(Path directory) throws IOException {
        return read(directory, Options.defaults());
    }

    public static PackageInspection read(Path directory, Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        TrainingReportPortfolioArtifactManifest.ManifestInspection manifest =
                TrainingReportPortfolioArtifactManifest.read(
                        resolvedDirectory,
                        resolvedOptions.manifest());
        TrainingReportPortfolioArtifacts.ArtifactInspection artifacts =
                TrainingReportPortfolioArtifactManifest.readArtifacts(manifest);
        return new PackageInspection(resolvedDirectory, manifest, artifacts);
    }

    public static PackageRefresh refreshComplete(Path directory) throws IOException {
        return refreshComplete(directory, Options.defaults(), Instant.now());
    }

    public static PackageRefresh refreshComplete(
            Path directory,
            Options options) throws IOException {
        return refreshComplete(directory, options, Instant.now());
    }

    public static PackageRefresh refreshComplete(
            Path directory,
            Options options,
            Instant generatedAt) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Instant resolvedGeneratedAt = Objects.requireNonNull(generatedAt, "generatedAt must not be null");

        TrainingReportPortfolioArtifactManifest.ManifestInspection manifest =
                TrainingReportPortfolioArtifactManifest.read(
                        resolvedDirectory,
                        resolvedOptions.manifest());
        verifyNonRefreshableManifestArtifacts(
                manifest,
                Set.of(
                        MARKDOWN_ARTIFACT,
                        LEADERBOARD_CSV_ARTIFACT,
                        COMPARISON_METRICS_CSV_ARTIFACT,
                        COMPARISON_FINDINGS_CSV_ARTIFACT));
        TrainingReportPortfolioArtifacts.ArtifactInspection artifacts =
                TrainingReportPortfolioArtifacts.refreshDerivedFiles(
                        requiredManifestArtifact(manifest, JSON_ARTIFACT).file(),
                        requiredManifestArtifact(manifest, MARKDOWN_ARTIFACT).file(),
                        requiredManifestArtifact(manifest, LEADERBOARD_CSV_ARTIFACT).file(),
                        requiredManifestArtifact(manifest, COMPARISON_METRICS_CSV_ARTIFACT).file(),
                        requiredManifestArtifact(manifest, COMPARISON_FINDINGS_CSV_ARTIFACT).file());
        TrainingReportPortfolioArtifactManifest.ManifestInspection refreshedManifest =
                TrainingReportPortfolioArtifactManifest.refresh(manifest, resolvedGeneratedAt);
        TrainingReportPortfolioArtifactManifest.ManifestVerification verification =
                verify(resolvedDirectory, null, resolvedOptions);
        TrainingReportPortfolioArtifactPackageReport.ReportBundle reports =
                TrainingReportPortfolioArtifactPackageReport.write(resolvedDirectory, verification);
        TrainingReportPortfolioArtifactPackageReport.ReportVerification reportVerification =
                TrainingReportPortfolioArtifactPackageReport.verify(reports);
        return new PackageRefresh(
                resolvedDirectory,
                refreshedManifest,
                artifacts,
                verification,
                reports,
                reportVerification);
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verify(
            PackageBundle bundle) throws IOException {
        Objects.requireNonNull(bundle, "bundle must not be null");
        return TrainingReportPortfolioArtifactManifest.verify(bundle.manifest());
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verify(
            Path directory) throws IOException {
        return verify(directory, null, Options.defaults());
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verify(
            Path directory,
            String expectedManifestSha256) throws IOException {
        return verify(directory, expectedManifestSha256, Options.defaults());
    }

    public static TrainingReportPortfolioArtifactManifest.ManifestVerification verify(
            Path directory,
            String expectedManifestSha256,
            Options options) throws IOException {
        Path resolvedDirectory = Objects.requireNonNull(directory, "directory must not be null")
                .toAbsolutePath()
                .normalize();
        Options resolvedOptions = options == null ? Options.defaults() : options;
        return TrainingReportPortfolioArtifactManifest.verify(
                resolvedDirectory,
                expectedManifestSha256,
                resolvedOptions.manifest());
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            Map<String, Path> reportFiles) throws IOException {
        PackageBundle bundle = write(directory, reportFiles);
        return new VerifiedPackageBundle(bundle, verify(bundle));
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            Map<String, Path> reportFiles,
            String baselineName) throws IOException {
        PackageBundle bundle = write(directory, reportFiles, baselineName);
        return new VerifiedPackageBundle(bundle, verify(bundle));
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPortfolioExport export) throws IOException {
        PackageBundle bundle = write(directory, export);
        return new VerifiedPackageBundle(bundle, verify(bundle));
    }

    public static VerifiedPackageBundle writeAndVerify(
            Path directory,
            TrainingReportPortfolioExport export,
            Options options,
            Instant generatedAt) throws IOException {
        PackageBundle bundle = write(directory, export, options, generatedAt);
        return new VerifiedPackageBundle(bundle, verify(bundle));
    }

    private static void verifyNonRefreshableManifestArtifacts(
            TrainingReportPortfolioArtifactManifest.ManifestInspection manifest,
            Set<String> refreshableArtifactNames) throws IOException {
        for (TrainingReportPortfolioArtifactManifest.ArtifactEntry artifact : manifest.artifacts().values()) {
            if (refreshableArtifactNames.contains(artifact.name())) {
                continue;
            }
            if (!Files.isRegularFile(artifact.file())) {
                throw new IOException("Cannot refresh portfolio package because non-refreshable artifact is missing: "
                        + artifact.name() + " (" + artifact.file() + ")");
            }
            TrainingReportArtifactFingerprint fingerprint = TrainingReportArtifactFingerprint.of(artifact.file());
            if (fingerprint.bytes() != artifact.bytes()) {
                throw new IOException("Cannot refresh portfolio package because non-refreshable artifact byte size "
                        + "changed: " + artifact.name() + " (" + artifact.file() + ")");
            }
            if (!artifact.sha256().equalsIgnoreCase(fingerprint.sha256())) {
                throw new IOException("Cannot refresh portfolio package because non-refreshable artifact SHA-256 "
                        + "changed: " + artifact.name() + " (" + artifact.file() + ")");
            }
        }
    }

    private static TrainingReportPortfolioArtifactManifest.ArtifactEntry requiredManifestArtifact(
            TrainingReportPortfolioArtifactManifest.ManifestInspection manifest,
            String artifactName) throws IOException {
        return manifest.artifact(artifactName)
                .orElseThrow(() -> new IOException(
                        "Cannot refresh portfolio package because manifest is missing required artifact: "
                                + artifactName));
    }
}
