package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.Options;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageInspection;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.PackageVerification;
import tech.kayys.gollek.ml.train.TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification;

final class TrainingReportPromotionGatePackageSupport {
    private static final String REVIEW_JSON_ARTIFACT = "reviewJson";
    private static final String REVIEW_MARKDOWN_ARTIFACT = "reviewMarkdown";

    private TrainingReportPromotionGatePackageSupport() {
    }

    static Map<String, Path> packageArtifactPaths(
            TrainingReportPromotionArtifacts.ArtifactBundle review,
            Map<String, Path> sourceSnapshots) {
        Map<String, Path> artifacts = new LinkedHashMap<>();
        artifacts.put(REVIEW_JSON_ARTIFACT, review.jsonFile());
        artifacts.put(REVIEW_MARKDOWN_ARTIFACT, review.markdownFile());
        artifacts.putAll(sourceSnapshots);
        return Map.copyOf(artifacts);
    }

    static void verifyNonRefreshableManifestArtifacts(
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            Set<String> refreshableArtifactNames) throws IOException {
        for (TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact : manifest.artifacts().values()) {
            if (refreshableArtifactNames.contains(artifact.name())) {
                continue;
            }
            TrainingReportPromotionGateArtifactPackageFingerprints.requireUnchangedNonRefreshableArtifact(artifact);
        }
    }

    static TrainingReportPromotionGateArtifactManifest.ArtifactEntry requiredManifestArtifact(
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            String artifactName) throws IOException {
        return manifest.artifact(artifactName)
                .orElseThrow(() -> new IOException(
                        "Cannot refresh promotion package because manifest is missing required artifact: "
                                + artifactName));
    }

    static PackageVerification packageVerification(
            PackageInspection inspection,
            TrainingReportPromotionGateArtifactManifest.ManifestVerification manifestVerification,
            SourceSnapshotVerification sourceSnapshotVerification) {
        List<String> failures = new ArrayList<>();
        failures.addAll(manifestVerification.failures());
        failures.addAll(sourceSnapshotVerification.failures());
        return new PackageVerification(
                inspection,
                manifestVerification,
                sourceSnapshotVerification,
                failures);
    }

    static PackageInspection readFromManifest(
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            Options options) throws IOException {
        TrainingReportPromotionGateArtifactManifest.ManifestInspection resolvedManifest =
                Objects.requireNonNull(manifest, "manifest must not be null");
        Options resolvedOptions = options == null ? Options.defaults() : options;
        Path directory = resolvedManifest.directory();
        TrainingReportPromotionArtifacts.ArtifactInspection review =
                readReviewArtifacts(directory, resolvedManifest, resolvedOptions.review());
        TrainingReportPromotionGateArtifacts.ArtifactInspection artifacts =
                TrainingReportPromotionGateArtifactManifest.readArtifacts(resolvedManifest);
        return new PackageInspection(directory, resolvedManifest, review, artifacts);
    }

    private static TrainingReportPromotionArtifacts.ArtifactInspection readReviewArtifacts(
            Path directory,
            TrainingReportPromotionGateArtifactManifest.ManifestInspection manifest,
            TrainingReportPromotionArtifacts.Options options) throws IOException {
        TrainingReportPromotionGateArtifactManifest.ArtifactEntry json =
                manifest.artifact(REVIEW_JSON_ARTIFACT).orElse(null);
        TrainingReportPromotionGateArtifactManifest.ArtifactEntry markdown =
                manifest.artifact(REVIEW_MARKDOWN_ARTIFACT).orElse(null);
        if (json != null && markdown != null) {
            return TrainingReportPromotionArtifacts.readFiles(json.file(), markdown.file());
        }
        return TrainingReportPromotionArtifacts.read(directory, options);
    }
}
