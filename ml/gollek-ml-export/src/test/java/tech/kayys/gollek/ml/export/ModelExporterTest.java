package tech.kayys.gollek.ml.export;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.gollek.gguf.core.GgmlType;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFParser;
import tech.kayys.gollek.gguf.loader.GGUFReader;
import tech.kayys.gollek.ml.nn.layer.Linear;

import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelExporterTest {

    @TempDir
    Path tempDir;

    @Test
    void toGGUFWritesParseableModelFile() throws Exception {
        Path output = tempDir.resolve("linear.gguf");
        Linear model = new Linear(2, 3);

        ModelExporter.builder()
                .model(model)
                .inputShape(1, 2)
                .metadata(Map.of(
                        "architecture", "gollek-linear",
                        "general.name", "linear-smoke",
                        "trainer.tags", List.of("java", "gguf")))
                .build()
                .toGGUF(output, ModelExporter.Quantization.NONE);

        assertTrue(Files.size(output) > 0);
        GGUFModel parsed = parse(output);
        assertEquals(3, parsed.version());
        assertEquals("gollek-linear", parsed.metadata().get("general.architecture"));
        assertEquals("linear-smoke", parsed.metadata().get("general.name"));
        assertEquals(2, parsed.tensors().size());
        assertTrue(parsed.tensors().stream().anyMatch(tensor -> tensor.name().equals("weight")));
        assertTrue(parsed.tensors().stream().anyMatch(tensor -> tensor.name().equals("bias")));
    }

    @Test
    void toGGUFInt4UsesQ4BlocksForAlignedTensors() throws Exception {
        Path output = tempDir.resolve("linear-q4.gguf");
        Linear model = new Linear(32, 1, false);

        ModelExporter.builder()
                .model(model)
                .inputShape(1, 32)
                .metadata(Map.of("architecture", "gollek-linear"))
                .build()
                .toGGUF(output, ModelExporter.Quantization.INT4);

        GGUFModel parsed = parse(output);
        assertEquals(1, parsed.tensors().size());
        assertEquals(GgmlType.Q4_0.id, parsed.tensors().getFirst().typeId());
        assertEquals(GgmlType.Q4_0.typeSize, parsed.tensors().getFirst().sizeInBytes());
    }

    private static GGUFModel parse(Path path) throws Exception {
        try (Arena arena = Arena.ofConfined(); GGUFReader reader = new GGUFReader(path, arena)) {
            return new GGUFParser().parse(reader.segment(), arena);
        }
    }
}
