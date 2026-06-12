package tech.kayys.gollek.ml.train;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Verifier for the JSON/Markdown/JUnit report bundle emitted for a promotion package.
 */
final class TrainingReportPromotionGateVerificationReportBundleVerifier {
    private TrainingReportPromotionGateVerificationReportBundleVerifier() {
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection read(Path reportDirectory)
            throws IOException {
        Path resolvedReportDirectory = Objects.requireNonNull(reportDirectory, "reportDirectory must not be null")
                .toAbsolutePath()
                .normalize();
        Path markdownFile = TrainingReportPromotionGateArtifactPackage.defaultVerificationMarkdownFile(
                resolvedReportDirectory);
        Path junitXmlFile = TrainingReportPromotionGateArtifactPackage.defaultVerificationJunitXmlFile(
                resolvedReportDirectory);
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection(
                resolvedReportDirectory,
                TrainingReportPromotionGateArtifactPackage.readVerificationReport(
                        TrainingReportPromotionGateArtifactPackage.defaultVerificationReportFile(
                                resolvedReportDirectory)),
                markdownFile,
                TrainingReportArtifactFingerprint.of(markdownFile).sha256(),
                junitXmlFile,
                TrainingReportArtifactFingerprint.of(junitXmlFile).sha256());
    }

    static TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification verify(
            Path reportDirectory,
            String expectedJsonReportSha256) throws IOException {
        TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection inspection =
                read(reportDirectory);
        TrainingReportPromotionGateArtifactPackage.VerificationReportVerification jsonVerification =
                TrainingReportPromotionGateArtifactPackage.verifyVerificationReport(
                        inspection.json().reportFile(),
                        expectedJsonReportSha256);
        List<String> failures = new ArrayList<>(jsonVerification.failures());
        boolean markdownMatchesRendered = false;
        boolean junitXmlMatchesRendered = false;
        TrainingReportPromotionGateArtifactPackage.PackageVerification packageVerification =
                jsonVerification.packageVerification();
        if (packageVerification == null) {
            failures.add("Verification report bundle cannot render derived reports because JSON package "
                    + "verification could not be rebuilt");
        } else {
            markdownMatchesRendered = verifyMarkdownReport(inspection, packageVerification, failures);
            junitXmlMatchesRendered = verifyJunitXmlReport(inspection, packageVerification, failures);
        }
        TrainingReportArtifactDescriptor.ChecksumMatch checksums =
                inspection.artifact().checksumMatch(expectedJsonReportSha256, null, null);
        return new TrainingReportPromotionGateArtifactPackage.VerificationReportBundleVerification(
                inspection,
                checksums.expectedJsonSha256(),
                jsonVerification.passed(),
                markdownMatchesRendered,
                junitXmlMatchesRendered,
                jsonVerification,
                failures);
    }

    private static boolean verifyMarkdownReport(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection inspection,
            TrainingReportPromotionGateArtifactPackage.PackageVerification packageVerification,
            List<String> failures) throws IOException {
        String expected = TrainingReportPromotionGateArtifactPackage.renderVerificationMarkdown(packageVerification);
        String actual = Files.readString(inspection.markdownFile(), StandardCharsets.UTF_8);
        if (!expected.equals(actual)) {
            failures.add("Verification report bundle markdown content mismatch for "
                    + inspection.markdownFile());
            return false;
        }
        return true;
    }

    private static boolean verifyJunitXmlReport(
            TrainingReportPromotionGateArtifactPackage.VerificationReportBundleInspection inspection,
            TrainingReportPromotionGateArtifactPackage.PackageVerification packageVerification,
            List<String> failures) throws IOException {
        String expected = TrainingReportPromotionGateArtifactPackage.renderVerificationJUnitXml(packageVerification);
        String actual = Files.readString(inspection.junitXmlFile(), StandardCharsets.UTF_8);
        if (!expected.equals(actual)) {
            failures.add("Verification report bundle JUnit XML content mismatch for "
                    + inspection.junitXmlFile());
            return false;
        }
        return true;
    }
}
