package tech.kayys.gollek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.converter.GGUFConverter;
import tech.kayys.gollek.converter.GGUFException;
import tech.kayys.gollek.converter.model.ConversionProgress;
import tech.kayys.gollek.converter.model.ConversionResult;
import tech.kayys.gollek.converter.model.GGUFConversionParams;
import tech.kayys.gollek.converter.model.QuantizationType;

import java.nio.file.Path;

@Dependent
@Unremovable
@Command(name = "convert", description = "Convert a model to GGUF (local)")
public class ConvertCommand implements Runnable {

    @Inject
    GGUFConverter converter;

    @Option(names = { "--input" }, description = "Input model path (file or directory)", required = true)
    String inputPath;

    @Option(names = { "--output" }, description = "Output GGUF file or directory", required = true)
    String outputPath;

    @Option(names = { "--quant" }, description = "Quantization type (e.g. q4_k_m, q8_0, f16)", defaultValue = "q4_k_m")
    String quantization;

    @Option(names = { "--model-type" }, description = "Model type hint (e.g. llama, mistral)")
    String modelType;

    @Option(names = { "--vocab-only" }, description = "Convert vocabulary only", defaultValue = "false")
    boolean vocabOnly;

    @Option(names = { "--threads" }, description = "Number of threads (0 = auto)", defaultValue = "0")
    int numThreads;

    @Option(names = { "--vocab-type" }, description = "Vocabulary type override (bpe, spm)")
    String vocabType;

    @Option(names = { "--pad-vocab" }, description = "Pad vocab to multiple", defaultValue = "0")
    int padVocab;

    @Option(names = { "--overwrite" }, description = "Overwrite output if it exists", defaultValue = "false")
    boolean overwriteExisting;

    @Option(names = { "--dry-run" }, description = "Resolve paths without converting", defaultValue = "false")
    boolean dryRun;

    @Option(names = { "--json" }, description = "Print JSON output (for scripting)", defaultValue = "false")
    boolean jsonOutput;

    @Option(names = { "--json-pretty" }, description = "Pretty-print JSON output", defaultValue = "false")
    boolean jsonPretty;

    @Option(names = { "--model-base" }, description = "Base path for relative input paths")
    String modelBase;

    @Option(names = { "--output-base" }, description = "Base path for relative output paths")
    String outputBase;

    @Override
    public void run() {
        try {
            applyBaseOverrides();
            if (jsonPretty) {
                jsonOutput = true;
            }
            QuantizationType quant = parseQuantization(quantization);
            GGUFConversionParams params = GGUFConversionParams.builder()
                    .inputPath(Path.of(inputPath))
                    .outputPath(Path.of(outputPath))
                    .modelType(modelType)
                    .quantization(quant)
                    .vocabOnly(vocabOnly)
                    .numThreads(numThreads)
                    .vocabType(vocabType)
                    .padVocab(padVocab)
                    .overwriteExisting(overwriteExisting)
                    .build();

            if (dryRun) {
                GGUFConversionParams resolved = converter.resolveParams(params);
                if (jsonOutput) {
                    printJsonResponse(true, true, resolved, null, null);
                } else {
                    System.out.println("Dry run resolved paths:");
                    System.out.println("  Input : " + resolved.getInputPath());
                    System.out.println("  Output: " + resolved.getOutputPath());
                }
                return;
            }

            if (!jsonOutput) {
                System.out.println("Converting model to GGUF...");
            }
            ConversionResult result = converter.convert(params, this::renderProgress);
            if (jsonOutput) {
                printJsonResponse(true, false, params, result, null);
            } else {
                System.out.println();
                System.out.println("Conversion complete:");
                System.out.println("  Output: " + result.getOutputPath());
                System.out.println("  Size  : " + result.getOutputSizeFormatted());
                System.out.println("  Time  : " + result.getDurationFormatted());
            }
        } catch (GGUFException e) {
            if (jsonOutput) {
                printJsonResponse(false, dryRun, null, null, e.getMessage());
            } else {
                System.err.println("Conversion failed: " + e.getMessage());
                if (e.getCause() != null) {
                    System.err.println("Caused by: " + e.getCause().getClass().getName() + ": " + e.getCause().getMessage());
                    e.getCause().printStackTrace(System.err);
                } else {
                    e.printStackTrace(System.err);
                }
            }
        } catch (Exception e) {
            if (jsonOutput) {
                printJsonResponse(false, dryRun, null, null, e.getMessage());
            } else {
                System.err.println("Conversion failed: " + e.getMessage());
                e.printStackTrace(System.err);
            }
        }
    }

    private void applyBaseOverrides() {
        if (modelBase != null && !modelBase.isBlank()) {
            System.setProperty("gollek.model.base", modelBase.trim());
        }
        if (outputBase != null && !outputBase.isBlank()) {
            System.setProperty("gollek.converter.base", outputBase.trim());
        }
    }

    private QuantizationType parseQuantization(String value) {
        if (value == null || value.isBlank()) {
            return QuantizationType.Q4_K_M;
        }
        String normalized = value.trim().toLowerCase();
        QuantizationType byNative = QuantizationType.fromNativeName(normalized);
        if (byNative != null) {
            return byNative;
        }
        try {
            return QuantizationType.valueOf(normalized.toUpperCase().replace('-', '_'));
        } catch (IllegalArgumentException e) {
            throw new GGUFException("Unknown quantization: " + value);
        }
    }

    private void renderProgress(ConversionProgress progress) {
        if (progress == null) {
            return;
        }
        if (jsonOutput) {
            return;
        }
        int percent = progress.getProgressPercent();
        String stage = progress.getStage() == null ? "Processing" : progress.getStage();
        System.out.printf("\r%3d%% - %s", percent, stage);
    }

    private void printJsonResponse(boolean success, boolean dryRunFlag, GGUFConversionParams resolvedParams,
            ConversionResult result, String errorMessage) {
        String inputPathValue = null;
        String outputPathValue = null;
        String derivedName = null;
        if (result != null) {
            outputPathValue = result.getOutputPath() == null ? null : result.getOutputPath().toString();
            derivedName = result.getOutputPath() == null ? null : result.getOutputPath().getFileName().toString();
        }
        if (resolvedParams != null) {
            if (inputPathValue == null) {
                inputPathValue = resolvedParams.getInputPath() == null ? null : resolvedParams.getInputPath().toString();
            }
            if (outputPathValue == null) {
                outputPathValue = resolvedParams.getOutputPath() == null ? null : resolvedParams.getOutputPath().toString();
            }
            if (derivedName == null) {
                derivedName = converter.deriveOutputName(resolvedParams.getInputPath(), resolvedParams.getQuantization());
            }
        }

        String inputBase = converter.resolveModelBasePath().toString();
        String outputBase = converter.resolveConverterBasePath().toString();

        java.util.List<String> fields = new java.util.ArrayList<>();
        fields.add(jsonField("success", String.valueOf(success), false));
        fields.add(jsonField("dryRun", String.valueOf(dryRunFlag), false));
        addJsonField(fields, "inputPath", inputPathValue, true);
        addJsonField(fields, "outputPath", outputPathValue, true);
        addJsonField(fields, "derivedOutputName", derivedName, true);
        addJsonField(fields, "inputBasePath", inputBase, true);
        addJsonField(fields, "outputBasePath", outputBase, true);
        if (result != null) {
            addJsonField(fields, "outputSize", result.getOutputSizeFormatted(), true);
            addJsonField(fields, "duration", result.getDurationFormatted(), true);
            fields.add(jsonField("compressionRatio", String.valueOf(result.getCompressionRatio()), false));
        }
        addJsonField(fields, "errorMessage", errorMessage, true);

        if (jsonPretty) {
            System.out.println("{\n  " + String.join(",\n  ", fields) + "\n}");
        } else {
            System.out.println("{" + String.join(",", fields) + "}");
        }
    }

    private void addJsonField(java.util.List<String> fields, String key, String value, boolean quote) {
        if (value == null) {
            return;
        }
        fields.add(jsonField(key, value, quote));
    }

    private String jsonField(String key, String value, boolean quote) {
        StringBuilder json = new StringBuilder();
        json.append('"').append(escapeJson(key)).append('"').append(':');
        if (quote) {
            json.append('"').append(escapeJson(value)).append('"');
        } else {
            json.append(value);
        }
        return json.toString();
    }

    private String escapeJson(String value) {
        StringBuilder escaped = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '\\':
                    escaped.append("\\\\");
                    break;
                case '"':
                    escaped.append("\\\"");
                    break;
                case '\n':
                    escaped.append("\\n");
                    break;
                case '\r':
                    escaped.append("\\r");
                    break;
                case '\t':
                    escaped.append("\\t");
                    break;
                default:
                    escaped.append(c);
            }
        }
        return escaped.toString();
    }
}
