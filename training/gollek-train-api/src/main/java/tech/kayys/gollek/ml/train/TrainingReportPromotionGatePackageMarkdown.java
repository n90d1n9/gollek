package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Objects;

/**
 * Markdown renderer for promotion gate package integrity verification.
 */
public final class TrainingReportPromotionGatePackageMarkdown {
    private TrainingReportPromotionGatePackageMarkdown() {
    }

    public static String render(TrainingReportPromotionGateArtifactPackage.PackageVerification verification) {
        TrainingReportPromotionGateArtifactPackage.PackageVerification resolvedVerification =
                Objects.requireNonNull(verification, "verification must not be null");
        TrainingReportPromotionGateArtifactPackage.PackageInspection inspection = resolvedVerification.inspection();
        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifest =
                resolvedVerification.manifestVerification();
        TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification sourceSnapshots =
                resolvedVerification.sourceSnapshotVerification();

        StringBuilder markdown = new StringBuilder();
        markdown.append("# Gollek Promotion Gate Package Verification\n\n");
        markdown.append("| Field | Value |\n");
        markdown.append("| --- | --- |\n");
        row(markdown, "Status", status(resolvedVerification.passed()));
        row(markdown, "Package", inspection.directory().toString());
        row(markdown, "Decision", inspection.manifest().decisionStatus());
        row(markdown, "Promotable", Boolean.toString(inspection.promotable()));
        inspection.manifest().decisionCandidate()
                .ifPresent(candidate -> row(markdown, "Candidate", candidate));
        row(markdown, "Manifest", status(manifest.passed()));
        row(markdown, "Manifest checksum", matchStatus(manifest.manifestSha256Matches()));
        row(markdown, "Artifact bytes", matchStatus(manifest.artifactBytesMatch()));
        row(markdown, "Artifact checksums", matchStatus(manifest.artifactSha256Match()));
        manifest.artifactVerificationOptional()
                .ifPresent(artifactVerification -> row(
                        markdown,
                        "JUnit XML",
                        artifactVerification.junitXmlWellFormed() ? "well formed" : "invalid"));
        row(markdown, "Source report snapshots", status(sourceSnapshots.passed()));
        row(markdown, "Expected source snapshots",
                Integer.toString(sourceSnapshots.expectedSourceReportArtifactNames().size()));
        row(markdown, "Present source snapshots",
                Integer.toString(sourceSnapshots.presentSourceReportArtifactNames().size()));
        markdown.append('\n');

        appendArtifactTable(markdown, inspection);
        appendSourceSnapshotSection(markdown, sourceSnapshots);
        appendFailures(markdown, resolvedVerification.failures());
        return markdown.toString();
    }

    private static void appendArtifactTable(
            StringBuilder markdown,
            TrainingReportPromotionGateArtifactPackage.PackageInspection inspection) {
        markdown.append("## Package Artifacts\n\n");
        markdown.append("| Artifact | Bytes | SHA-256 |\n");
        markdown.append("| --- | ---: | --- |\n");
        for (TrainingReportPromotionGateArtifactManifest.ArtifactEntry artifact
                : inspection.manifest().artifacts().values()) {
            markdown.append("| ")
                    .append(cell(artifact.name()))
                    .append(" | ")
                    .append(artifact.bytes())
                    .append(" | `")
                    .append(artifact.sha256())
                    .append("` |\n");
        }
        markdown.append('\n');
    }

    private static void appendSourceSnapshotSection(
            StringBuilder markdown,
            TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification sourceSnapshots) {
        markdown.append("## Source Report Snapshots\n\n");
        if (sourceSnapshots.snapshots().isEmpty()) {
            markdown.append("No source report snapshots were packaged.\n\n");
        } else {
            markdown.append("| Role | Name | Artifact | SHA-256 |\n");
            markdown.append("| --- | --- | --- | --- |\n");
            for (TrainingReportPromotionGateArtifactPackage.SourceReportSnapshot snapshot : sourceSnapshots.snapshots()) {
                markdown.append("| ")
                        .append(cell(snapshot.role()))
                        .append(" | ")
                        .append(cell(snapshot.name()))
                        .append(" | ")
                        .append(cell(snapshot.artifact().name()))
                        .append(" | `")
                        .append(snapshot.artifact().sha256())
                        .append("` |\n");
            }
            markdown.append('\n');
        }
        appendNamedList(markdown, "Missing snapshots", sourceSnapshots.missingSourceReportArtifactNames());
        appendNamedList(markdown, "Unexpected snapshots", sourceSnapshots.unexpectedSourceReportArtifactNames());
    }

    private static void appendFailures(StringBuilder markdown, List<String> failures) {
        markdown.append("## Failures\n\n");
        if (failures.isEmpty()) {
            markdown.append("None.\n");
            return;
        }
        for (String failure : failures) {
            markdown.append("- ").append(line(failure)).append('\n');
        }
    }

    private static void appendNamedList(StringBuilder markdown, String title, List<String> values) {
        markdown.append("### ").append(title).append("\n\n");
        if (values.isEmpty()) {
            markdown.append("None.\n\n");
            return;
        }
        for (String value : values) {
            markdown.append("- `").append(value).append("`\n");
        }
        markdown.append('\n');
    }

    private static void row(StringBuilder markdown, String field, String value) {
        markdown.append("| ")
                .append(cell(field))
                .append(" | ")
                .append(cell(value))
                .append(" |\n");
    }

    private static String status(boolean passed) {
        return passed ? "PASS" : "FAIL";
    }

    private static String matchStatus(boolean matched) {
        return matched ? "match" : "mismatch";
    }

    private static String cell(String value) {
        return line(value).replace("|", "\\|");
    }

    private static String line(String value) {
        return value == null ? "" : value.replace("\r", " ").replace("\n", " ").trim();
    }
}
