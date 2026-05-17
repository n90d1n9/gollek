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

        assertEquals("gemma4", profile.architecture());
        assertEquals(11, profile.requiredDecoderTensorCount());
        assertEquals(11, profile.presentDecoderTensorCount());
        assertEquals(1.0d, profile.knownTensorTypeRatio(), 0.0001d);
        assertTrue(profile.decoderTensorSetComplete());
        assertTrue(profile.rowDotPrimitivesReady());
        assertEquals("loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled",
                profile.javaStatus());
        assertTrue(profile.compactTypeSummary(2).contains("Q4_K"));
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

    private static GGUFTensorInfo tensor(String name, int typeId, long sizeInBytes) {
        return new GGUFTensorInfo(name, new long[]{1}, typeId, 0, sizeInBytes);
    }
}
