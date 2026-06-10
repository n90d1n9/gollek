///usr/bin/env jbang "$0" "$@" ; exit $?
//NOINTEGRATIONS
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS -Dquarkus.jbang.post-build=false --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-backend-metal:0.1.0-SNAPSHOT

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.layer.Linear;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGateManifest;
import tech.kayys.gollek.ml.train.TrainingReportQualityProfileCiGateManifestVerificationReport;
import tech.kayys.gollek.trainer.api.TrainingSummary;

public class trainer_quality_profile_ci_gate_evidence {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Options options = Options.parse(args);
        Files.createDirectories(options.outputRoot());

        System.out.println("====================================================");
        System.out.println(" Gollek Quality Profile CI Gate Evidence (JBang)");
        System.out.println("====================================================");
        System.out.println("outputRoot=" + options.outputRoot().toAbsolutePath().normalize());
        System.out.println("profile=" + options.profileId());
        System.out.println("device=" + options.device());
        System.out.println("backend=" + Gollek.DL.accelerationStatus(options.device()));

        Path baselineReport = trainTinyClassifier(
                "baseline",
                options.outputRoot().resolve("runs").resolve("baseline"),
                options.baselineEpochs(),
                2026L,
                options.device());
        Path candidateReport = trainTinyClassifier(
                "candidate",
                options.outputRoot().resolve("runs").resolve("candidate"),
                options.candidateEpochs(),
                2027L,
                options.device());

        Map<String, Path> reports = new LinkedHashMap<>();
        reports.put("baseline", baselineReport);
        reports.put("candidate", candidateReport);

        Path gateDir = options.outputRoot().resolve("quality-profile-ci-gate");
        var gate = Gollek.DL.runTrainingReportQualityProfileCiGate(
                reports,
                "baseline",
                options.profileId(),
                gateDir);
        TrainingReportQualityProfileCiGateManifest.ManifestBundle manifest =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifest(gateDir, gate);
        TrainingReportQualityProfileCiGateManifest.ManifestVerification manifestVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifest(manifest);
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportBundle reportBundle =
                Gollek.DL.writeTrainingReportQualityProfileCiGateManifestVerificationReport(
                        gateDir.resolve("manifest-verification-report"),
                        manifestVerification);
        TrainingReportQualityProfileCiGateManifestVerificationReport.ReportVerification reportVerification =
                Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReport(
                        reportBundle.directory(),
                        manifestVerification);
        var receipt = Gollek.DL.writeTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                reportBundle.directory(),
                reportVerification,
                "quality-profile-ci-gate-manifest-verification.receipt.json");
        var receiptVerification = Gollek.DL.verifyTrainingReportQualityProfileCiGateManifestVerificationReportReceipt(
                receipt.receiptFile(),
                receipt.receiptSha256(),
                manifestVerification);

        System.out.println("----------------------------------------------------");
        System.out.println("baselineReport=" + baselineReport);
        System.out.println("candidateReport=" + candidateReport);
        System.out.println("gatePassed=" + gate.passed());
        System.out.println("gateMessage=" + gate.message());
        System.out.println("manifestVerified=" + manifestVerification.passed());
        System.out.println("manifestReadyForRelease=" + manifestVerification.summary().readyForRelease());
        System.out.println("manifestFailedCategories=" + manifestVerification.summary().failedCategories());
        System.out.println("verificationReportVerified=" + reportVerification.passed());
        System.out.println("verificationReportJson=" + reportBundle.jsonFile());
        System.out.println("verificationReportMarkdown=" + reportBundle.markdownFile());
        System.out.println("verificationReportJunitXml=" + reportBundle.junitXmlReport().junitXmlFile());
        System.out.println("verificationReportSha256=" + reportBundle.jsonSha256());
        System.out.println("verificationReportReceipt=" + receipt.receiptFile());
        System.out.println("verificationReportReceiptSha256=" + receipt.receiptSha256());
        System.out.println("verificationReportReceiptVerified=" + receiptVerification.passed());

        if (!reportVerification.passed()) {
            reportVerification.failures().forEach(failure -> System.out.println("verificationFailure=" + failure));
            System.exit(3);
        }
        if (!receiptVerification.passed()) {
            receiptVerification.failures().forEach(failure -> System.out.println("receiptFailure=" + failure));
            System.exit(4);
        }
        if (options.failOnGate() && !gate.passed()) {
            System.exit(2);
        }
    }

    private static Path trainTinyClassifier(
            String name,
            Path checkpointDir,
            int epochs,
            long seed,
            String device) throws Exception {
        Linear model = new Linear(3, 3);
        var split = Gollek.DL.classificationStratifiedTrainValidationSplit(
                demoInputs(),
                new int[] {0, 0, 1, 1, 2, 2, 0, 1, 2},
                0.67,
                seed);
        TrainingSummary summary = Gollek.DL.fit(
                model,
                split,
                3,
                true,
                seed,
                epochs,
                0.08f,
                Gollek.DL.TrainingPreset.CLASSIFICATION_CROSS_ENTROPY_SGD,
                Gollek.DL.trainingOptions()
                        .device(device)
                        .gradientClip(1.0)
                        .classificationMetrics()
                        .checkpointDir(checkpointDir)
                        .build());
        Path reportFile = checkpointDir.resolve("canonical-report.json");
        if (!Files.isRegularFile(reportFile)) {
            throw new IllegalStateException("Training report was not written for " + name + ": " + reportFile
                    + " metadata=" + summary.metadata());
        }
        System.out.printf(
                "%s epochs=%d trainLoss=%.6f validationLoss=%.6f report=%s%n",
                name,
                summary.epochCount(),
                summary.latestTrainLoss(),
                summary.latestValidationLoss(),
                reportFile);
        return reportFile;
    }

    private static GradTensor demoInputs() {
        return GradTensor.of(new float[] {
                1f, 0f, 0f,
                0.9f, 0.1f, 0f,
                0f, 1f, 0f,
                0.1f, 0.9f, 0f,
                0f, 0f, 1f,
                0f, 0.2f, 0.8f,
                0.8f, 0f, 0.2f,
                0.2f, 0.7f, 0.1f,
                0.1f, 0.1f, 0.8f
        }, 9, 3);
    }

    private record Options(
            Path outputRoot,
            String profileId,
            String device,
            int baselineEpochs,
            int candidateEpochs,
            boolean failOnGate) {
        static Options parse(String[] args) {
            Path outputRoot = Path.of("trainer_checkpoints", "quality_profile_ci_gate_evidence");
            String profileId = "local-experiment";
            String device = "auto";
            int baselineEpochs = 2;
            int candidateEpochs = 8;
            boolean failOnGate = false;
            for (int index = 0; index < args.length; index++) {
                String arg = args[index];
                switch (arg) {
                    case "--out" -> outputRoot = Path.of(requireValue(args, ++index, arg));
                    case "--profile" -> profileId = requireValue(args, ++index, arg);
                    case "--device" -> device = requireValue(args, ++index, arg);
                    case "--baseline-epochs" -> baselineEpochs = parsePositiveInt(requireValue(args, ++index, arg), arg);
                    case "--candidate-epochs" -> candidateEpochs = parsePositiveInt(requireValue(args, ++index, arg), arg);
                    case "--fail-on-gate" -> failOnGate = true;
                    default -> {
                        if (!arg.isBlank()) {
                            outputRoot = Path.of(arg);
                        }
                    }
                }
            }
            return new Options(
                    outputRoot.toAbsolutePath().normalize(),
                    profileId,
                    device,
                    baselineEpochs,
                    candidateEpochs,
                    failOnGate);
        }

        private static String requireValue(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException(flag + " requires a value");
            }
            return args[index];
        }

        private static int parsePositiveInt(String value, String flag) {
            try {
                return Math.max(1, Integer.parseInt(value.trim()));
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException(flag + " requires a positive integer: " + value, error);
            }
        }
    }
}
