package tech.kayys.gollek.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufRuntimeProfileTest {

    @Test
    void reportsCompleteDecoderTensorCoverageForKnownLayouts() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "gemma4");
        metadata.put("gemma4.block_count", 1);
        metadata.put("gemma4.embedding_length", 2304);
        metadata.put("gemma4.attention.head_count", 8);
        metadata.put("gemma4.attention.head_count_kv", 4);

        List<GGUFTensorInfo> tensors = new ArrayList<>();
        tensors.add(tensor("token_embd.weight", 0, 16, 2304, 256));
        tensors.add(tensor("output_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.attn_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.attn_q.weight", 12, 144, 2304, 2304));
        tensors.add(tensor("blk.0.attn_k.weight", 12, 144, 2304, 1152));
        tensors.add(tensor("blk.0.attn_v.weight", 12, 144, 2304, 1152));
        tensors.add(tensor("blk.0.attn_q_norm.weight", 0, 16, 288));
        tensors.add(tensor("blk.0.attn_k_norm.weight", 0, 16, 288));
        tensors.add(tensor("blk.0.attn_output.weight", 12, 144, 2304, 2304));
        tensors.add(tensor("blk.0.ffn_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.ffn_gate.weight", 12, 144, 2304, 9216));
        tensors.add(tensor("blk.0.ffn_up.weight", 12, 144, 2304, 9216));
        tensors.add(tensor("blk.0.ffn_down.weight", 12, 144, 9216, 2304));

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1280, 7);

        assertEquals("gemma4", profile.architecture());
        assertEquals(13, profile.requiredDecoderTensorCount());
        assertEquals(13, profile.presentDecoderTensorCount());
        assertEquals(0, profile.missingDecoderTensorCount());
        assertEquals(0, profile.malformedDecoderTensorCount());
        assertEquals(1.0d, profile.knownTensorTypeRatio(), 0.0001d);
        assertTrue(profile.decoderTensorSetComplete());
        assertTrue(profile.rowDotPrimitivesReady());
        assertEquals("loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled",
                profile.javaStatus());
        assertTrue(profile.compactTypeSummary(2).contains("Q4_K"));
        assertEquals("gemma4", profile.modelConfig().modelType());
        assertEquals(2304, profile.modelConfig().hiddenSize());
        assertEquals(1, profile.modelConfig().numHiddenLayers());
        assertEquals(8, profile.modelConfig().numAttentionHeads());
        assertEquals(4, profile.modelConfig().resolvedNumKvHeads());
    }

    @Test
    void requiresGemma4AttentionNormTensorsBeforeDeclaringDecoderReady() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "gemma4");
        metadata.put("gemma4.block_count", 1);

        List<GGUFTensorInfo> tensors = new ArrayList<>();
        tensors.add(tensor("token_embd.weight", 0, 16));
        tensors.add(tensor("output_norm.weight", 0, 16));
        tensors.add(tensor("blk.0.attn_norm.weight", 0, 16));
        tensors.add(tensor("blk.0.attn_q.weight", 12, 144));
        tensors.add(tensor("blk.0.attn_k.weight", 12, 144));
        tensors.add(tensor("blk.0.attn_v.weight", 12, 144));
        tensors.add(tensor("blk.0.attn_output.weight", 12, 144));
        tensors.add(tensor("blk.0.ffn_norm.weight", 0, 16));
        tensors.add(tensor("blk.0.ffn_gate.weight", 12, 144));
        tensors.add(tensor("blk.0.ffn_up.weight", 12, 144));
        tensors.add(tensor("blk.0.ffn_down.weight", 12, 144));

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1280, 7);

        assertEquals(13, profile.requiredDecoderTensorCount());
        assertEquals(11, profile.presentDecoderTensorCount());
        assertEquals(2, profile.missingDecoderTensorCount());
        assertEquals(0, profile.malformedDecoderTensorCount());
        assertEquals("loader-ready; decoder-tensors-missing=2; generation-disabled", profile.javaStatus());
        assertTrue(profile.missingDecoderTensorExamples().contains("blk.0.attn_q_norm.weight"));
        assertTrue(profile.missingDecoderTensorExamples().contains("blk.0.attn_k_norm.weight"));
    }

    @Test
    void treatsQ2KAndQ3KDecoderTensorsAsJavaRowDotReady() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.block_count", 1);

        List<GGUFTensorInfo> tensors = List.of(
                tensor("token_embd.weight", 10, 84, 256, 256),
                tensor("output_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_q.weight", 10, 84, 256, 256),
                tensor("blk.0.attn_k.weight", 11, 110, 256, 256),
                tensor("blk.0.attn_v.weight", 10, 84, 256, 256),
                tensor("blk.0.attn_output.weight", 11, 110, 256, 256),
                tensor("blk.0.ffn_norm.weight", 0, 16, 256),
                tensor("blk.0.ffn_gate.weight", 10, 84, 256, 256),
                tensor("blk.0.ffn_up.weight", 11, 110, 256, 256),
                tensor("blk.0.ffn_down.weight", 10, 84, 256, 256)
        );

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1024, 3);

        assertEquals(11, profile.requiredDecoderTensorCount());
        assertEquals(11, profile.presentDecoderTensorCount());
        assertTrue(profile.unsupportedRowDotTypeIds().isEmpty());
        assertTrue(profile.rowDotPrimitivesReady());
        assertEquals("loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled",
                profile.javaStatus());
    }

    @Test
    void treatsIQ4DecoderTensorsAsJavaRowDotReady() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.block_count", 1);

        List<GGUFTensorInfo> tensors = List.of(
                tensor("token_embd.weight", 20, 18, 32, 256),
                tensor("output_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_q.weight", 23, 136, 256, 256),
                tensor("blk.0.attn_k.weight", 20, 18, 32, 256),
                tensor("blk.0.attn_v.weight", 23, 136, 256, 256),
                tensor("blk.0.attn_output.weight", 23, 136, 256, 256),
                tensor("blk.0.ffn_norm.weight", 0, 16, 256),
                tensor("blk.0.ffn_gate.weight", 23, 136, 256, 256),
                tensor("blk.0.ffn_up.weight", 23, 136, 256, 256),
                tensor("blk.0.ffn_down.weight", 23, 136, 256, 256)
        );

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1024, 3);

        assertEquals(11, profile.requiredDecoderTensorCount());
        assertEquals(11, profile.presentDecoderTensorCount());
        assertTrue(profile.unsupportedRowDotTypeIds().isEmpty());
        assertTrue(profile.rowDotPrimitivesReady());
        assertTrue(profile.compactTypeSummary(3).contains("IQ4_XS"));
        assertEquals("loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled",
                profile.javaStatus());
    }

    @Test
    void treatsQ8_1DecoderTensorsAsJavaRowDotReady() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.block_count", 1);

        List<GGUFTensorInfo> tensors = List.of(
                tensor("token_embd.weight", 9, 36, 32, 256),
                tensor("output_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_norm.weight", 0, 16, 256),
                tensor("blk.0.attn_q.weight", 9, 36, 256, 256),
                tensor("blk.0.attn_k.weight", 9, 36, 256, 256),
                tensor("blk.0.attn_v.weight", 9, 36, 256, 256),
                tensor("blk.0.attn_output.weight", 9, 36, 256, 256),
                tensor("blk.0.ffn_norm.weight", 0, 16, 256),
                tensor("blk.0.ffn_gate.weight", 9, 36, 256, 256),
                tensor("blk.0.ffn_up.weight", 9, 36, 256, 256),
                tensor("blk.0.ffn_down.weight", 9, 36, 256, 256)
        );

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1024, 3);

        assertEquals(11, profile.requiredDecoderTensorCount());
        assertEquals(11, profile.presentDecoderTensorCount());
        assertTrue(profile.unknownTensorTypeIds().isEmpty());
        assertTrue(profile.unsupportedRowDotTypeIds().isEmpty());
        assertTrue(profile.rowDotPrimitivesReady());
        assertTrue(profile.compactTypeSummary(3).contains("Q8_1"));
        assertEquals("loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled",
                profile.javaStatus());
    }

    @Test
    void rejectsDecoderProjectionShapeDriftWhenMetadataProvidesDimensions() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "gemma4");
        metadata.put("gemma4.block_count", 1);
        metadata.put("gemma4.embedding_length", 2304);
        metadata.put("gemma4.attention.head_count", 8);
        metadata.put("gemma4.attention.head_count_kv", 4);

        List<GGUFTensorInfo> tensors = new ArrayList<>();
        tensors.add(tensor("token_embd.weight", 0, 16, 2304, 256));
        tensors.add(tensor("output_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.attn_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.attn_q.weight", 12, 144, 1024, 2304));
        tensors.add(tensor("blk.0.attn_k.weight", 12, 144, 2304, 1152));
        tensors.add(tensor("blk.0.attn_v.weight", 12, 144, 2304, 1152));
        tensors.add(tensor("blk.0.attn_q_norm.weight", 0, 16, 288));
        tensors.add(tensor("blk.0.attn_k_norm.weight", 0, 16, 288));
        tensors.add(tensor("blk.0.attn_output.weight", 12, 144, 2304, 2304));
        tensors.add(tensor("blk.0.ffn_norm.weight", 0, 16, 2304));
        tensors.add(tensor("blk.0.ffn_gate.weight", 12, 144, 2304, 9216));
        tensors.add(tensor("blk.0.ffn_up.weight", 12, 144, 2304, 9216));
        tensors.add(tensor("blk.0.ffn_down.weight", 12, 144, 9216, 2304));

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 1280, 7);

        assertEquals(13, profile.requiredDecoderTensorCount());
        assertEquals(12, profile.presentDecoderTensorCount());
        assertEquals(0, profile.missingDecoderTensorCount());
        assertEquals(1, profile.malformedDecoderTensorCount());
        assertEquals("loader-ready; decoder-tensor-shapes-invalid=1; generation-disabled", profile.javaStatus());
        assertTrue(profile.missingDecoderTensorExamples().isEmpty());
        assertTrue(profile.malformedDecoderTensorExamples().stream()
                .anyMatch(example -> example.startsWith("blk.0.attn_q.weight shape=")));
    }

    @Test
    void reportsMissingDecoderTensorsAndUnknownTypes() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.block_count", 1);

        List<GGUFTensorInfo> tensors = List.of(
                tensor("token_embd.weight", 0, 16),
                tensor("output_norm.weight", 0, 16),
                tensor("blk.0.attn_norm.weight", 99, 16)
        );

        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model(metadata, tensors), 64, 1);

        assertEquals(11, profile.requiredDecoderTensorCount());
        assertEquals(3, profile.presentDecoderTensorCount());
        assertTrue(profile.knownTensorTypeRatio() < 1.0d);
        assertEquals(List.of("99"), profile.unknownTensorTypeIds());
        assertEquals(List.of("99"), profile.unsupportedRowDotTypeIds());
        assertEquals("loader-ready; unknown-tensor-types=99; generation-disabled", profile.javaStatus());
        assertTrue(profile.missingDecoderTensorExamples().contains("blk.0.attn_q.weight"));
    }

    private static GGUFModel model(Map<String, Object> metadata, List<GGUFTensorInfo> tensors) {
        return new GGUFModel(3, metadata, tensors, 0, MemorySegment.NULL, null);
    }

    private static GGUFTensorInfo tensor(String name, int typeId, long sizeInBytes, long... shape) {
        long[] tensorShape = shape.length > 0 ? shape : new long[]{1};
        return new GGUFTensorInfo(name, tensorShape, typeId, 0, sizeInBytes);
    }
}
