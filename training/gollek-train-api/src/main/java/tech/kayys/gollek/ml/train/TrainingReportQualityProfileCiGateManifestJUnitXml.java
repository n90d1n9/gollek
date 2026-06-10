package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * JUnit XML renderer for quality-profile CI manifest verification.
 */
public final class TrainingReportQualityProfileCiGateManifestJUnitXml {
    public static final String SUITE_NAME = "gollek.training.quality-profile.ci-gate.manifest";

    private TrainingReportQualityProfileCiGateManifestJUnitXml() {
    }

    public record Report(
            Path junitXmlFile,
            String junitXml,
            String junitXmlSha256,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        public Report {
            junitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                    .toAbsolutePath()
                    .normalize();
            junitXml = Objects.requireNonNull(junitXml, "junitXml must not be null");
            junitXmlSha256 = requireChecksum(junitXmlSha256, "junitXmlSha256");
            verification = Objects.requireNonNull(verification, "verification must not be null");
        }

        public boolean passed() {
            return verification.passed();
        }

        public boolean wellFormed() {
            return TrainingReportXml.isWellFormed(junitXml);
        }

        public Map<String, Object> toMap() {
            return Map.of(
                    "junitXmlFile", junitXmlFile.toString(),
                    "junitXmlSha256", junitXmlSha256,
                    "passed", passed(),
                    "wellFormed", wellFormed(),
                    "verification", verification.toMap());
        }
    }

    public static String render(TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        Objects.requireNonNull(verification, "verification must not be null");
        List<Check> checks = checks(verification);
        StringBuilder xml = new StringBuilder();
        appendLine(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        appendLine(xml, "<testsuite name=\"" + SUITE_NAME + "\" tests=\"" + checks.size()
                + "\" failures=\"" + checks.stream().filter(check -> !check.passed()).count()
                + "\" errors=\"0\" skipped=\"0\">");
        appendProperties(xml, verification);
        for (Check check : checks) {
            appendCheck(xml, check);
        }
        appendLine(xml, "</testsuite>");
        return xml.toString();
    }

    public static Report write(
            Path junitXmlFile,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) throws IOException {
        Path resolvedJunitXmlFile = Objects.requireNonNull(junitXmlFile, "junitXmlFile must not be null")
                .toAbsolutePath()
                .normalize();
        String junitXml = render(verification);
        TrainerCheckpointIO.writeStringAtomically(resolvedJunitXmlFile, junitXml);
        return new Report(
                resolvedJunitXmlFile,
                junitXml,
                TrainingReportArtifactFingerprint.of(resolvedJunitXmlFile).sha256(),
                verification);
    }

    private static List<Check> checks(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        Map<String, Boolean> outcomes = checkOutcomes(verification);
        return List.of(
                new Check(
                        "checksums",
                        outcomes.get("checksums"),
                        verification.checksumFailures()),
                new Check("format", outcomes.get("format"), verification.formatFailures()),
                new Check("profile", outcomes.get("profile"), verification.profileFailures()),
                new Check("markdown", outcomes.get("markdown"), verification.markdownFailures()),
                new Check("structure", outcomes.get("structure"), verification.structureFailures()),
                new Check("artifacts", outcomes.get("artifacts"), verification.artifactFailures()));
    }

    static Map<String, Boolean> checkOutcomes(
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        Objects.requireNonNull(verification, "verification must not be null");
        Map<String, Boolean> outcomes = new LinkedHashMap<>();
        outcomes.put("checksums", verification.jsonSha256Matches() && verification.markdownSha256Matches());
        outcomes.put("format", verification.formatValid());
        outcomes.put("profile", verification.profileKnown());
        outcomes.put("markdown", verification.markdownMatchesJson());
        outcomes.put("structure", verification.structureValid());
        outcomes.put("artifacts", verification.artifactsMatch());
        return Collections.unmodifiableMap(outcomes);
    }

    private static void appendProperties(
            StringBuilder xml,
            TrainingReportQualityProfileCiGateManifest.ManifestVerification verification) {
        TrainingReportQualityProfileCiGateManifestSummary summary = verification.summary();
        appendLine(xml, "  <properties>");
        property(xml, "manifest.passed", Boolean.toString(verification.passed()));
        property(xml, "manifest.status", summary.status());
        property(xml, "manifest.readyForRelease", Boolean.toString(summary.readyForRelease()));
        property(xml, "manifest.structureValid", Boolean.toString(verification.structureValid()));
        property(xml, "manifest.artifactsMatch", Boolean.toString(verification.artifactsMatch()));
        property(xml, "manifest.markdownMatchesJson", Boolean.toString(verification.markdownMatchesJson()));
        property(xml, "manifest.formatValid", Boolean.toString(verification.formatValid()));
        property(xml, "manifest.profileKnown", Boolean.toString(verification.profileKnown()));
        property(xml, "manifest.failureCount", Integer.toString(summary.failureCount()));
        property(xml, "manifest.failedCategoryCount", Integer.toString(summary.failedCategories().size()));
        summary.primaryFailureCategory().ifPresent(value -> property(xml, "manifest.primaryFailureCategory", value));
        Map<TrainingReportQualityProfileCiGateManifestFailureCategory, Integer> counts =
                summary.failureCountsByCategory();
        for (TrainingReportQualityProfileCiGateManifestFailureCategory category
                : TrainingReportQualityProfileCiGateManifestFailureCategory.values()) {
            property(xml, "manifest.failures." + category.id(), Integer.toString(counts.getOrDefault(category, 0)));
        }
        appendLine(xml, "  </properties>");
    }

    private static void appendCheck(StringBuilder xml, Check check) {
        appendLine(xml, "  <testcase classname=\"" + SUITE_NAME
                + "\" name=\"" + escapeXml(check.name()) + "\" time=\"0\">");
        if (!check.passed()) {
            String message = check.failures().isEmpty()
                    ? "Manifest " + check.name() + " verification failed"
                    : check.failures().get(0);
            appendLine(xml, "    <failure type=\"MANIFEST_" + check.name().toUpperCase().replace('-', '_')
                    + "_FAILED\" message=\"" + escapeXml(message) + "\">");
            appendLine(xml, escapeText(String.join("\n", check.failures())));
            appendLine(xml, "    </failure>");
        }
        appendLine(xml, "  </testcase>");
    }

    private static void property(StringBuilder xml, String name, String value) {
        appendLine(xml, "    <property name=\"" + escapeXml(name)
                + "\" value=\"" + escapeXml(value) + "\"/>");
    }

    private static String requireChecksum(String checksum, String fieldName) {
        if (checksum == null || checksum.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return checksum.trim().toLowerCase();
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

    private record Check(String name, boolean passed, List<String> failures) {
        private Check {
            failures = failures == null ? List.of() : List.copyOf(failures);
        }
    }
}
