///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek SDK Export & Benchmark Example
//
// Demonstrates model export and benchmarking with:
// - Export to ONNX, GGUF, LiteRT formats
// - Performance benchmarking
// - Quantization options
//
// Prerequisites:
//   - Java 25+
//   - JBang: curl -Ls https://sh.jbang.dev | bash -s -
//   - All SDK modules built
//
// Usage:
//   jbang gollek-sdk-export-example.java
//
// DEPS tech.kayys.gollek:gollek-sdk-export:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-sdk-vision:0.1.0-SNAPSHOT
// JAVA 25+

import tech.kayys.gollek.sdk.export.*;
import tech.kayys.gollek.sdk.vision.models.ResNet;

import java.nio.file.Path;

public class gollek_sdk_export_example {

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║     Gollek SDK Export & Benchmark Example                ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // 1. Create model
        System.out.println("1. Creating ResNet-18 model...");
        ResNet model = ResNet.resnet18(1000);
        System.out.println("   ✓ Model: " + model);
        System.out.println();

        // 2. Create output directory
        Path outputDir = Path.of("exported_models");
        java.nio.file.Files.createDirectories(outputDir);
        System.out.println("2. Output directory: " + outputDir.toAbsolutePath());
        System.out.println();

        // 3. Create exporter
        System.out.println("3. Creating model exporter...");
        ModelExporter exporter = ModelExporter.builder()
            .model(model)
            .inputShape(1, 3, 224, 224)
            .build();
        System.out.println("   ✓ Exporter configured");
        System.out.println("   ✓ Input shape: [1, 3, 224, 224]");
        System.out.println();

        // 4. Export to ONNX
        System.out.println("4. Exporting to ONNX...");
        exporter.toONNX(outputDir.resolve("model.onnx"));
        System.out.println("   ✓ Saved: exported_models/model.onnx");
        System.out.println();

        // 5. Export to GGUF with different quantizations
        System.out.println("5. Exporting to GGUF with quantization...");

        System.out.println("   a) FP16 quantization...");
        exporter.toGGUF(outputDir.resolve("model_fp16.gguf"), ModelExporter.Quantization.FP16);
        System.out.println("      ✓ Saved: exported_models/model_fp16.gguf");

        System.out.println("   b) INT8 quantization...");
        exporter.toGGUF(outputDir.resolve("model_int8.gguf"), ModelExporter.Quantization.INT8);
        System.out.println("      ✓ Saved: exported_models/model_int8.gguf");

        System.out.println("   c) INT4 quantization...");
        exporter.toGGUF(outputDir.resolve("model_int4.gguf"), ModelExporter.Quantization.INT4);
        System.out.println("      ✓ Saved: exported_models/model_int4.gguf");

        System.out.println("   d) NF4 quantization (QLoRA-style)...");
        exporter.toGGUF(outputDir.resolve("model_nf4.gguf"), ModelExporter.Quantization.NF4);
        System.out.println("      ✓ Saved: exported_models/model_nf4.gguf");
        System.out.println();

        // 6. Export to LiteRT
        System.out.println("6. Exporting to LiteRT (edge format)...");
        exporter.toLiteRT(outputDir.resolve("model.litert"));
        System.out.println("   ✓ Saved: exported_models/model.litert");
        System.out.println();

        // 7. Benchmark
        System.out.println("7. Running performance benchmark...");
        Benchmark benchmark = new Benchmark(model);
        BenchmarkResult result = benchmark.run(new long[]{1, 3, 224, 224}, 100, 10);

        System.out.println();
        System.out.println("   Benchmark Results:");
        System.out.println("   ┌─────────────────────────────────────────┐");
        System.out.printf("   │ %-35s │%n", "Metric");
        System.out.println("   ├─────────────────────────────────────────┤");
        System.out.printf("   │ %-35s │%n", "Avg Latency: " + String.format("%.2f ms", result.avgLatencyMs()));
        System.out.printf("   │ %-35s │%n", "P50 Latency: " + String.format("%.2f ms", result.p50LatencyMs()));
        System.out.printf("   │ %-35s │%n", "P95 Latency: " + String.format("%.2f ms", result.p95LatencyMs()));
        System.out.printf("   │ %-35s │%n", "P99 Latency: " + String.format("%.2f ms", result.p99LatencyMs()));
        System.out.printf("   │ %-35s │%n", "Throughput: " + String.format("%.1f inf/s", result.throughput()));
        System.out.printf("   │ %-35s │%n", "Iterations: " + result.iterations());
        System.out.println("   └─────────────────────────────────────────┘");
        System.out.println();

        // 8. Summary
        System.out.println("8. Export summary:");
        System.out.println("   ┌──────────────────────────────────────────────────────┐");
        System.out.println("   │ Format  │ Quantization │ File Size (est.) │ Use Case │");
        System.out.println("   ├──────────────────────────────────────────────────────┤");
        System.out.println("   │ ONNX    │ FP32         │ ~45 MB           │ General  │");
        System.out.println("   │ GGUF    │ FP16         │ ~22 MB           │ GPU      │");
        System.out.println("   │ GGUF    │ INT8         │ ~11 MB           │ CPU      │");
        System.out.println("   │ GGUF    │ INT4         │ ~5.5 MB          │ Edge     │");
        System.out.println("   │ GGUF    │ NF4          │ ~5.5 MB          │ QLoRA    │");
        System.out.println("   │ LiteRT  │ FP32         │ ~45 MB           │ Mobile   │");
        System.out.println("   └──────────────────────────────────────────────────────┘");
        System.out.println();

        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║  Export & Benchmark example complete!                    ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
    }
}
