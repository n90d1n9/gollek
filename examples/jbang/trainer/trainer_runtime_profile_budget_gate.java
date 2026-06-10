///usr/bin/env jbang "$0" "$@" ; exit $?
//NOINTEGRATIONS
//JAVA 25
//JAVAC_OPTIONS --release 25 --enable-preview --add-modules=jdk.incubator.vector
//JAVA_OPTIONS -Dquarkus.jbang.post-build=false --enable-preview --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-api:0.1.0-SNAPSHOT

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.ml.train.TrainingReport;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGate;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeInputProfileGateArtifacts;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileBudgetGate;
import tech.kayys.gollek.ml.train.TrainingReportRuntimeProfileBudgetGateArtifacts;

public class trainer_runtime_profile_budget_gate {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        Options options = Options.parse(args);
        Files.createDirectories(options.outputRoot());

        TrainingReport report = TrainingReport.of(canonicalReportPayload());
        Path reportFile = options.outputRoot().resolve("canonical-runtime-profile-report.json");
        Files.writeString(reportFile, Json.write(report.payload()) + "\n");

        TrainingReportRuntimeProfileBudgetGate.Result result =
                Gollek.DL.trainingReportRuntimeProfileBudgetGate(report, options.policy());
        TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactBundle bundle =
                Gollek.DL.writeTrainingReportRuntimeProfileBudgetGateArtifacts(
                        options.outputRoot().resolve("runtime-profile-budget-gate"),
                        result);
        TrainingReportRuntimeProfileBudgetGateArtifacts.ArtifactVerification verification =
                Gollek.DL.verifyTrainingReportRuntimeProfileBudgetGateArtifacts(bundle);
        TrainingReportRuntimeInputProfileGate.Result inputResult =
                Gollek.DL.trainingReportRuntimeInputProfileGate(report, TrainingReportRuntimeInputProfileGate.Policy.strict());
        TrainingReportRuntimeInputProfileGateArtifacts.ArtifactBundle inputBundle =
                Gollek.DL.writeTrainingReportRuntimeInputProfileGateArtifacts(
                        options.outputRoot().resolve("runtime-input-profile-gate"),
                        inputResult);
        TrainingReportRuntimeInputProfileGateArtifacts.ArtifactVerification inputVerification =
                Gollek.DL.verifyTrainingReportRuntimeInputProfileGateArtifacts(inputBundle);

        System.out.println("====================================================");
        System.out.println(" Gollek Runtime Profile Budget Gate (JBang)");
        System.out.println("====================================================");
        System.out.println("outputRoot=" + options.outputRoot());
        System.out.println("reportFile=" + reportFile);
        System.out.println("available=" + result.available());
        System.out.println("passed=" + result.passed());
        System.out.println("message=" + result.message());
        System.out.println("findingCount=" + result.findings().size());
        result.findings().forEach(finding -> System.out.printf(
                "- %s severity=%s action=%s evidence=%s%n",
                finding.code(),
                finding.severity(),
                finding.action(),
                finding.evidence()));
        System.out.println("artifactDir=" + bundle.directory());
        System.out.println("artifactJson=" + bundle.jsonFile());
        System.out.println("artifactMarkdown=" + bundle.markdownFile());
        System.out.println("artifactJunitXml=" + bundle.junitXmlFile());
        System.out.println("artifactJsonSha256=" + bundle.jsonSha256());
        System.out.println("artifactsVerified=" + verification.passed());
        System.out.println("verificationMessage=" + verification.message());
        System.out.println("inputGatePassed=" + inputResult.passed());
        System.out.println("inputGateFindingCount=" + inputResult.findings().size());
        inputResult.findings().forEach(finding -> System.out.printf(
                "- input %s severity=%s action=%s evidence=%s%n",
                finding.code(),
                finding.severity(),
                finding.action(),
                finding.evidence()));
        System.out.println("inputArtifactDir=" + inputBundle.directory());
        System.out.println("inputArtifactJson=" + inputBundle.jsonFile());
        System.out.println("inputArtifactMarkdown=" + inputBundle.markdownFile());
        System.out.println("inputArtifactJunitXml=" + inputBundle.junitXmlFile());
        System.out.println("inputArtifactsVerified=" + inputVerification.passed());
        System.out.println("inputVerificationMessage=" + inputVerification.message());
        System.out.println("----------------------------------------------------");
        System.out.println(result.markdown());
        System.out.println("----------------------------------------------------");
        System.out.println(inputResult.markdown());

        verification.requirePassed();
        inputVerification.requirePassed();
        if (options.failOnGate()) {
            result.requirePassed();
            inputResult.requirePassed();
        }
    }

    private static Map<String, Object> canonicalReportPayload() {
        Map<String, Object> metadata = runtimeMetadata();
        List<Map<String, Object>> history = List.of(Map.of(
                "epoch", 0,
                "trainLoss", 0.7,
                "validationLoss", 0.8,
                "learningRate", 0.01));
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema", "gollek.training.report.v1");
        payload.put("generatedAt", Instant.parse("2026-05-26T11:12:13Z").toString());
        payload.put("epochCount", 1);
        payload.put("bestValidationLoss", 0.8);
        payload.put("bestValidationEpoch", 0);
        payload.put("latestTrainLoss", 0.7);
        payload.put("latestValidationLoss", 0.8);
        payload.put("durationMs", 100L);
        payload.put("history", history);
        payload.put("historySummary", Map.of("rows", history.size()));
        payload.put("metadata", metadata);
        return payload;
    }

    private static Map<String, Object> runtimeMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("epochHistory", List.of(Map.of(
                "epoch", 0,
                "trainLoss", 0.7,
                "validationLoss", 0.8,
                "learningRate", 0.01)));
        metadata.put("runtimeProfile.groupCount", 1);
        metadata.put("runtimeProfile.primaryGroup.name", "input");
        metadata.put("runtimeProfile.primaryGroup.totalMillis", 500.0);
        metadata.put("runtimeProfile.primaryGroup.percentTotal", 86.0);
        metadata.put("runtimeProfile.groups", List.of(Map.of(
                "name", "input",
                "count", 12L,
                "totalMillis", 500.0,
                "percentTotal", 86.0,
                "averageMillis", 41.667)));
        metadata.put("runtimeProfile.hotspotCount", 1);
        metadata.put("runtimeProfile.primaryHotspot.phase", "input.train.next");
        metadata.put("runtimeProfile.primaryHotspot.totalMillis", 320.0);
        metadata.put("runtimeProfile.primaryHotspot.percentTotal", 72.0);
        metadata.put("runtimeProfile.hotspots", List.of(Map.of(
                "phase", "input.train.next",
                "count", 4L,
                "totalMillis", 320.0,
                "percentTotal", 72.0,
                "averageMillis", 80.0)));
        metadata.put("runtimeProfile.input.train.iterator.totalMillis", 10.0);
        metadata.put("runtimeProfile.input.train.hasNext.totalMillis", 20.0);
        metadata.put("runtimeProfile.input.train.next.count", 4L);
        metadata.put("runtimeProfile.input.train.next.totalMillis", 320.0);
        metadata.put("runtimeProfile.input.validation.next.totalMillis", 15.0);
        metadata.put("trainLoaderPlan.batchCount", 4);
        metadata.put("trainLoaderPlan.prefetch.enabled", false);
        metadata.put("trainLoaderPlan.prefetch.maxBufferedItems", 0);
        metadata.put("trainLoaderPlan.prefetch.summary", "prefetch[enabled=false]");
        return Map.copyOf(metadata);
    }

    private record Options(
            Path outputRoot,
            TrainingReportRuntimeProfileBudgetGate.Policy policy,
            boolean failOnGate) {
        static Options parse(String[] args) {
            Path outputRoot = Path.of("trainer_checkpoints", "runtime_profile_budget_gate");
            TrainingReportRuntimeProfileBudgetGate.Policy policy =
                    new TrainingReportRuntimeProfileBudgetGate.Policy(80.0, 60.0, 250.0);
            boolean failOnGate = false;
            for (int index = 0; index < args.length; index++) {
                String arg = args[index];
                switch (arg) {
                    case "--out" -> outputRoot = Path.of(requireValue(args, ++index, arg));
                    case "--policy" -> policy = namedPolicy(requireValue(args, ++index, arg));
                    case "--max-primary-group-percent" ->
                            policy = policy.withMaxPrimaryGroupPercent(parseDouble(requireValue(args, ++index, arg), arg));
                    case "--max-primary-hotspot-percent" ->
                            policy = policy.withMaxPrimaryHotspotPercent(parseDouble(requireValue(args, ++index, arg), arg));
                    case "--max-primary-hotspot-ms" ->
                            policy = policy.withMaxPrimaryHotspotTotalMillis(parseDouble(requireValue(args, ++index, arg), arg));
                    case "--fail-on-gate" -> failOnGate = true;
                    default -> {
                        if (!arg.isBlank()) {
                            outputRoot = Path.of(arg);
                        }
                    }
                }
            }
            return new Options(outputRoot.toAbsolutePath().normalize(), policy, failOnGate);
        }

        private static TrainingReportRuntimeProfileBudgetGate.Policy namedPolicy(String value) {
            return switch (value.trim().toLowerCase(Locale.ROOT)) {
                case "default", "defaults" -> TrainingReportRuntimeProfileBudgetGate.Policy.defaults()
                        .withMaxPrimaryHotspotTotalMillis(250.0);
                case "strict" -> TrainingReportRuntimeProfileBudgetGate.Policy.strict();
                case "permissive" -> TrainingReportRuntimeProfileBudgetGate.Policy.permissive();
                default -> throw new IllegalArgumentException("Unknown policy: " + value);
            };
        }

        private static String requireValue(String[] args, int index, String flag) {
            if (index >= args.length) {
                throw new IllegalArgumentException(flag + " requires a value");
            }
            return args[index];
        }

        private static double parseDouble(String value, String flag) {
            try {
                return Double.parseDouble(value.trim());
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException(flag + " requires a number: " + value, error);
            }
        }
    }

    private static final class Json {
        private Json() {
        }

        static String write(Object value) {
            if (value == null) {
                return "null";
            }
            if (value instanceof String text) {
                return "\"" + escape(text) + "\"";
            }
            if (value instanceof Number || value instanceof Boolean) {
                return String.valueOf(value);
            }
            if (value instanceof Map<?, ?> map) {
                StringBuilder json = new StringBuilder("{");
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) {
                        json.append(',');
                    }
                    first = false;
                    json.append(write(String.valueOf(entry.getKey()))).append(':').append(write(entry.getValue()));
                }
                return json.append('}').toString();
            }
            if (value instanceof Iterable<?> iterable) {
                StringBuilder json = new StringBuilder("[");
                boolean first = true;
                for (Object item : iterable) {
                    if (!first) {
                        json.append(',');
                    }
                    first = false;
                    json.append(write(item));
                }
                return json.append(']').toString();
            }
            return write(String.valueOf(value));
        }

        private static String escape(String value) {
            StringBuilder escaped = new StringBuilder();
            for (int index = 0; index < value.length(); index++) {
                char c = value.charAt(index);
                switch (c) {
                    case '"' -> escaped.append("\\\"");
                    case '\\' -> escaped.append("\\\\");
                    case '\n' -> escaped.append("\\n");
                    case '\r' -> escaped.append("\\r");
                    case '\t' -> escaped.append("\\t");
                    default -> {
                        if (c < 0x20) {
                            escaped.append(String.format("\\u%04x", (int) c));
                        } else {
                            escaped.append(c);
                        }
                    }
                }
            }
            return escaped.toString();
        }
    }
}
