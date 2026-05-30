package tech.kayys.gollek.ml.train;

import java.util.List;
import java.util.Objects;

/**
 * JUnit XML renderer for promotion gate package integrity verification.
 */
public final class TrainingReportPromotionGatePackageJUnitXml {
    private TrainingReportPromotionGatePackageJUnitXml() {
    }

    public static String render(TrainingReportPromotionGateArtifactPackage.PackageVerification verification) {
        TrainingReportPromotionGateArtifactPackage.PackageVerification resolvedVerification =
                Objects.requireNonNull(verification, "verification must not be null");
        TrainingReportPromotionGateArtifactPackage.PackageInspection inspection = resolvedVerification.inspection();
        TrainingReportPromotionGateArtifactManifest.ManifestVerification manifest =
                resolvedVerification.manifestVerification();
        TrainingReportPromotionGateArtifactPackage.SourceSnapshotVerification sourceSnapshots =
                resolvedVerification.sourceSnapshotVerification();
        String markdown = TrainingReportPromotionGateArtifactPackage.renderVerificationMarkdown(resolvedVerification);

        int failures = (manifest.passed() ? 0 : 1)
                + (sourceSnapshots.passed() ? 0 : 1)
                + (resolvedVerification.passed() ? 0 : 1);

        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"gollek.training.promotion.package\" tests=\"3\" failures=\""
                + failures + "\" errors=\"0\" skipped=\"0\">");
        appendLine(xml, "  <properties>");
        property(xml, "package.directory", inspection.directory().toString());
        property(xml, "package.passed", Boolean.toString(resolvedVerification.passed()));
        property(xml, "gate.passed", Boolean.toString(inspection.passed()));
        property(xml, "gate.promotable", Boolean.toString(inspection.promotable()));
        property(xml, "gate.decision.status", inspection.manifest().decisionStatus());
        inspection.manifest().decisionCandidate()
                .ifPresent(candidate -> property(xml, "gate.decision.candidate", candidate));
        property(xml, "manifest.file", inspection.manifest().manifestFile().toString());
        property(xml, "manifest.sha256", inspection.manifest().manifestSha256());
        property(xml, "manifest.verified", Boolean.toString(manifest.passed()));
        property(xml, "manifest.checksumMatches", Boolean.toString(manifest.manifestSha256Matches()));
        property(xml, "artifacts.bytesMatch", Boolean.toString(manifest.artifactBytesMatch()));
        property(xml, "artifacts.sha256Match", Boolean.toString(manifest.artifactSha256Match()));
        property(xml, "sourceSnapshots.verified", Boolean.toString(sourceSnapshots.passed()));
        property(xml, "sourceSnapshots.expected",
                Integer.toString(sourceSnapshots.expectedSourceReportArtifactNames().size()));
        property(xml, "sourceSnapshots.present",
                Integer.toString(sourceSnapshots.presentSourceReportArtifactNames().size()));
        property(xml, "sourceSnapshots.missing",
                Integer.toString(sourceSnapshots.missingSourceReportArtifactNames().size()));
        property(xml, "sourceSnapshots.unexpected",
                Integer.toString(sourceSnapshots.unexpectedSourceReportArtifactNames().size()));
        appendLine(xml, "  </properties>");

        testcase(
                xml,
                "manifest",
                manifest.passed(),
                "MANIFEST_VERIFICATION",
                manifest.message(),
                manifest.failures(),
                markdown);
        testcase(
                xml,
                "source-report-snapshots",
                sourceSnapshots.passed(),
                "SOURCE_REPORT_SNAPSHOT_VERIFICATION",
                sourceSnapshots.message(),
                sourceSnapshots.failures(),
                markdown);
        testcase(
                xml,
                "complete-package",
                resolvedVerification.passed(),
                "PACKAGE_VERIFICATION",
                resolvedVerification.message(),
                resolvedVerification.failures(),
                markdown);

        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    private static void testcase(
            StringBuilder xml,
            String name,
            boolean passed,
            String failureType,
            String message,
            List<String> failures,
            String markdown) {
        appendLine(xml, "  <testcase classname=\"gollek.training.promotion.package\" name=\""
                + escapeXml(name) + "\" time=\"0\">");
        if (!passed) {
            appendLine(xml, "    <failure type=\"" + escapeXml(failureType)
                    + "\" message=\"" + escapeXml(message) + "\">");
            appendLine(xml, escapeText(String.join("\n", failures)));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "    <system-out>" + escapeText(markdown) + "</system-out>");
        appendLine(xml, "  </testcase>");
    }

    private static void property(StringBuilder xml, String name, String value) {
        appendLine(xml, "    <property name=\"" + escapeXml(name)
                + "\" value=\"" + escapeXml(value) + "\"/>");
    }

    private static String escapeXml(String value) {
        return escapeText(value).replace("\"", "&quot;").replace("'", "&apos;");
    }

    private static String escapeText(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        StringBuilder escaped = new StringBuilder(value.length());
        for (int index = 0; index < value.length(); index++) {
            char ch = value.charAt(index);
            switch (ch) {
                case '&' -> escaped.append("&amp;");
                case '<' -> escaped.append("&lt;");
                case '>' -> escaped.append("&gt;");
                default -> {
                    if (isValidXmlChar(ch)) {
                        escaped.append(ch);
                    }
                }
            }
        }
        return escaped.toString();
    }

    private static boolean isValidXmlChar(char ch) {
        return ch == 0x9
                || ch == 0xA
                || ch == 0xD
                || (ch >= 0x20 && ch <= 0xD7FF)
                || (ch >= 0xE000 && ch <= 0xFFFD);
    }

    private static void appendLine(StringBuilder builder, String line) {
        builder.append(line).append('\n');
    }
}
