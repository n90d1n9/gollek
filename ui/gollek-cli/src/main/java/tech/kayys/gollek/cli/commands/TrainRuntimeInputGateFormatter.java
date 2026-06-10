package tech.kayys.gollek.cli.commands;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGate;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGateArtifacts;

/**
 * Formats runtime input-profile gate results and artifact verification summaries for CLI users.
 */
final class TrainRuntimeInputGateFormatter {
    private TrainRuntimeInputGateFormatter() {
    }

    static void printGateSummary(
            Path reportFile,
            TrainingReportRuntimeInputProfileGate.Result result,
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts) {
        System.out.println("Runtime input-profile gate: " + status(result.passed()));
        System.out.println("Report: " + reportFile.toAbsolutePath().normalize());
        System.out.println("Available: " + result.available());
        System.out.println("Findings: " + result.findings().size());
        System.out.println(result.message());
        for (TrainingReportRuntimeInputProfileGate.Finding finding : result.findings()) {
            TrainRuntimeInputGateFindingFormatter.print(finding);
        }
        if (artifacts != null) {
            System.out.println("Artifacts:");
            printArtifactFiles(artifacts);
        }
    }

    static void printVerificationSummary(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification,
            String heading) {
        if (heading != null) {
            System.out.println(heading + ": " + status(verification.passed()));
        }
        System.out.println("Directory: " + verification.inspection().directory());
        System.out.println("Gate passed: " + status(verification.inspection().passed()));
        printVerificationChecks(verification);
        printFailuresOrMessage(verification);
    }

    static void printAutoVerificationSummary(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification) {
        System.out.println("Training artifact verification: " + status(verification.passed()));
        System.out.println("Type: runtime-input-gate");
        printVerificationSummary(verification, null);
    }

    static void printRefreshSummary(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts,
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification) {
        System.out.println("Training artifacts refreshed: " + status(verification.passed()));
        System.out.println("Type: runtime-input-gate");
        System.out.println("Directory: " + artifacts.directory());
        printArtifactFiles(artifacts);
        System.out.println("Gate passed: " + status(artifacts.passed()));
        System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
        System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
        System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
        if (!verification.passed()) {
            for (String failure : verification.failures()) {
                System.out.println("- " + failure);
            }
        }
    }

    static Map<String, Object> resultPayload(
            TrainingReportRuntimeInputProfileGate.Result result,
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("result", result.toMap());
        if (artifacts != null) {
            payload.put("artifacts", artifacts.toMap());
        }
        return Map.copyOf(payload);
    }

    static Map<String, Object> artifactPayload(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts,
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("type", "runtime-input-gate");
        payload.put("artifacts", artifacts.toMap());
        payload.put("verification", verification.toMap());
        return Map.copyOf(payload);
    }

    private static void printArtifactFiles(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle artifacts) {
        System.out.println("- json: " + artifacts.jsonFile() + " sha256=" + artifacts.jsonSha256());
        System.out.println("- markdown: " + artifacts.markdownFile() + " sha256=" + artifacts.markdownSha256());
        System.out.println("- junitXml: " + artifacts.junitXmlFile() + " sha256=" + artifacts.junitXmlSha256());
    }

    private static void printVerificationChecks(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification) {
        System.out.println("JSON checksum: " + status(verification.jsonSha256Matches()));
        System.out.println("Markdown checksum: " + status(verification.markdownSha256Matches()));
        System.out.println("JUnit XML checksum: " + status(verification.junitXmlSha256Matches()));
        System.out.println("JUnit XML well-formed: " + status(verification.junitXmlWellFormed()));
        System.out.println("Markdown matches JSON: " + status(verification.markdownMatchesJson()));
        System.out.println("JUnit XML matches JSON: " + status(verification.junitXmlMatchesJson()));
    }

    private static void printFailuresOrMessage(
            TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification verification) {
        if (!verification.passed()) {
            for (String failure : verification.failures()) {
                System.out.println("- " + failure);
            }
        } else {
            System.out.println(verification.message());
        }
    }

    private static String status(boolean passed) {
        return passed ? "PASS" : "FAIL";
    }
}
