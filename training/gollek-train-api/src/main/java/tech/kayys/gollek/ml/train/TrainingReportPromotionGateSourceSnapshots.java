package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageInspection;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.SourceReportSnapshot;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification;

final class TrainingReportPromotionGateSourceSnapshots {
    private static final String ARTIFACT_PREFIX = "sourceReport.";

    private TrainingReportPromotionGateSourceSnapshots() {
    }

    static List<SourceReportSnapshot> snapshots(PackageInspection inspection) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<SourceReportSnapshot> snapshots = new ArrayList<>();
        Set<String> artifactNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.review().sourceReports()) {
            String artifactName = artifactName(report, artifactNames);
            inspection.manifest().artifact(artifactName)
                    .ifPresent(artifact -> snapshots.add(new SourceReportSnapshot(report, artifact)));
            artifactNames.add(artifactName);
        }
        return List.copyOf(snapshots);
    }

    static List<String> expectedArtifactNames(PackageInspection inspection) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        Set<String> artifactNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.review().sourceReports()) {
            artifactNames.add(artifactName(report, artifactNames));
        }
        return List.copyOf(artifactNames);
    }

    static List<String> presentArtifactNames(PackageInspection inspection) {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<String> artifactNames = new ArrayList<>();
        for (String artifactName : inspection.manifest().artifacts().keySet()) {
            if (artifactName.startsWith(ARTIFACT_PREFIX)) {
                artifactNames.add(artifactName);
            }
        }
        return List.copyOf(artifactNames);
    }

    static List<String> missingArtifactNames(PackageInspection inspection) {
        Set<String> present = new LinkedHashSet<>(presentArtifactNames(inspection));
        List<String> missing = new ArrayList<>();
        for (String artifactName : expectedArtifactNames(inspection)) {
            if (!present.contains(artifactName)) {
                missing.add(artifactName);
            }
        }
        return List.copyOf(missing);
    }

    static List<String> unexpectedArtifactNames(PackageInspection inspection) {
        Set<String> expected = new LinkedHashSet<>(expectedArtifactNames(inspection));
        List<String> unexpected = new ArrayList<>();
        for (String artifactName : presentArtifactNames(inspection)) {
            if (!expected.contains(artifactName)) {
                unexpected.add(artifactName);
            }
        }
        return List.copyOf(unexpected);
    }

    static SourceSnapshotVerification verify(PackageInspection inspection) throws IOException {
        Objects.requireNonNull(inspection, "inspection must not be null");
        List<SourceReportSnapshot> snapshots = new ArrayList<>();
        List<String> failures = new ArrayList<>();
        Set<String> artifactNames = new LinkedHashSet<>();
        Set<String> expectedArtifactNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.review().sourceReports()) {
            String artifactName = artifactName(report, artifactNames);
            artifactNames.add(artifactName);
            expectedArtifactNames.add(artifactName);
            TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact =
                    inspection.manifest().artifact(artifactName).orElse(null);
            if (artifact == null) {
                failures.add("Missing packaged source report snapshot for "
                        + report.role() + " " + report.name() + " (" + artifactName + ")");
                continue;
            }
            SourceReportSnapshot snapshot = new SourceReportSnapshot(report, artifact);
            snapshots.add(snapshot);
            TrainingReportPromotionGateArtifactPackageFingerprints.verifySourceReportSnapshot(snapshot, failures);
        }
        verifyNoUnexpectedArtifacts(inspection, expectedArtifactNames, failures);
        return new SourceSnapshotVerification(inspection, snapshots, failures);
    }

    static Map<String, Path> snapshotFiles(
            Path directory,
            TrainingReportPromotionArtifacts.ArtifactBundle review) throws IOException {
        TrainingReportPromotionArtifacts.ArtifactInspection inspection =
                TrainingReportPromotionArtifacts.readFiles(review.jsonFile(), review.markdownFile());
        Map<String, Path> snapshots = new LinkedHashMap<>();
        Set<String> fileNames = new LinkedHashSet<>();
        for (TrainingReportPromotionArtifacts.SourceReport report : inspection.sourceReports()) {
            if (report.source() == null || !Files.isRegularFile(report.source())) {
                continue;
            }
            String artifactName = artifactName(report, snapshots.keySet());
            String fileName = fileName(report, fileNames);
            Path target = directory.resolve(fileName).toAbsolutePath().normalize();
            if (!Files.exists(target) || !Files.isSameFile(report.source(), target)) {
                Files.copy(report.source(), target, StandardCopyOption.REPLACE_EXISTING);
            }
            snapshots.put(artifactName, target);
            fileNames.add(fileName);
        }
        return Map.copyOf(snapshots);
    }

    private static void verifyNoUnexpectedArtifacts(
            PackageInspection inspection,
            Set<String> expectedArtifactNames,
            List<String> failures) {
        for (String artifactName : inspection.manifest().artifacts().keySet()) {
            if (!artifactName.startsWith(ARTIFACT_PREFIX)
                    || expectedArtifactNames.contains(artifactName)) {
                continue;
            }
            failures.add("Unexpected packaged source report snapshot in manifest: " + artifactName);
        }
    }

    private static String artifactName(
            TrainingReportPromotionArtifacts.SourceReport report,
            Set<String> existing) {
        String base = ARTIFACT_PREFIX + safeToken(report.role()) + "." + safeToken(report.name());
        String name = base;
        int suffix = 2;
        while (existing.contains(name)) {
            name = base + "." + suffix++;
        }
        return name;
    }

    private static String fileName(
            TrainingReportPromotionArtifacts.SourceReport report,
            Set<String> existing) {
        String base = "source-report-" + safeToken(report.role()) + "-" + safeToken(report.name());
        String suffix = fileSuffix(report);
        String fileName = base + suffix;
        int duplicate = 2;
        while (existing.contains(fileName)) {
            fileName = base + "-" + duplicate++ + suffix;
        }
        return fileName;
    }

    private static String fileSuffix(TrainingReportPromotionArtifacts.SourceReport report) {
        String sha = report.sha256();
        String fingerprint = sha == null || sha.isBlank() ? "" : "-" + sha.substring(0, Math.min(12, sha.length()));
        String extension = fileExtension(report.source());
        return fingerprint + extension;
    }

    private static String fileExtension(Path source) {
        if (source == null || source.getFileName() == null) {
            return ".json";
        }
        String fileName = source.getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        if (dot <= 0 || dot == fileName.length() - 1) {
            return ".json";
        }
        String extension = fileName.substring(dot);
        return extension.length() > 16 ? ".json" : extension;
    }

    private static String safeToken(String value) {
        String text = value == null ? "report" : value.trim().toLowerCase(java.util.Locale.ROOT);
        StringBuilder token = new StringBuilder();
        for (int index = 0; index < text.length(); index++) {
            char ch = text.charAt(index);
            if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) {
                token.append(ch);
            } else if (token.length() > 0 && token.charAt(token.length() - 1) != '-') {
                token.append('-');
            }
        }
        while (token.length() > 0 && token.charAt(token.length() - 1) == '-') {
            token.setLength(token.length() - 1);
        }
        return token.length() == 0 ? "report" : token.toString();
    }
}
