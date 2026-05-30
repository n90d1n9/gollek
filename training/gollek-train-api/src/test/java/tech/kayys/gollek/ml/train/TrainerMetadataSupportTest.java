package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.Parameter;

class TrainerMetadataSupportTest {

    @Test
    void detectsPresentFilesOnly() throws Exception {
        Path file = Files.createTempFile("gollek-trainer-metadata", ".txt");
        Path missing = file.resolveSibling(file.getFileName() + ".missing");

        assertTrue(TrainerMetadataSupport.filePresent(file));
        assertFalse(TrainerMetadataSupport.filePresent(missing));
        assertFalse(TrainerMetadataSupport.filePresent(null));
    }

    @Test
    void normalizesDeviceIds() {
        assertEquals("auto", TrainerMetadataSupport.normalizeDevice(null));
        assertEquals("auto", TrainerMetadataSupport.normalizeDevice("   "));
        assertEquals("auto", TrainerMetadataSupport.normalizeDevice(" GPU "));
        assertEquals("metal", TrainerMetadataSupport.normalizeDevice(" Metal_GPU "));
        assertEquals("metal", TrainerMetadataSupport.normalizeDevice("mps"));
        assertEquals("cuda", TrainerMetadataSupport.normalizeDevice("nvidia"));
        assertEquals("cpu", TrainerMetadataSupport.normalizeDevice("off"));
    }

    @Test
    void readsIntegerCheckpointValuesSafely() {
        assertEquals(7, TrainerMetadataSupport.readInt(7L, 1));
        assertEquals(8, TrainerMetadataSupport.readInt("8", 1));
        assertEquals(1, TrainerMetadataSupport.readInt("not-int", 1));
        assertEquals(1, TrainerMetadataSupport.readInt(null, 1));
    }

    @Test
    void flattensMetricsIntoTargetMap() {
        Map<String, Object> target = new LinkedHashMap<>();

        TrainerMetadataSupport.flatten(target, "trainMetric.", Map.of("mae", 0.5));
        TrainerMetadataSupport.flatten(target, "detail.", Map.of("confusion", Map.of("tp", 2L)));

        assertEquals(0.5, target.get("trainMetric.mae"));
        assertEquals(Map.of("tp", 2L), target.get("detail.confusion"));
    }

    @Test
    void stateSnapshotIsDeepImmutableCopy() {
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("values", List.of(1, 2));
        Map<String, Object> source = new LinkedHashMap<>();
        source.put("nested", nested);

        Map<String, Object> snapshot = TrainerMetadataSupport.stateSnapshot(source);
        nested.put("after", "mutation");

        @SuppressWarnings("unchecked")
        Map<String, Object> snapshotNested = (Map<String, Object>) snapshot.get("nested");
        assertEquals(Map.of("values", List.of(1, 2)), snapshotNested);
        assertThrows(UnsupportedOperationException.class, () -> snapshot.put("x", 1));
        assertThrows(UnsupportedOperationException.class, () -> snapshotNested.put("x", 1));
    }

    @Test
    void parameterSignaturesAreStableForNamedAndIndexedParameters() {
        Parameter first = new Parameter(GradTensor.of(new float[] {1f, 2f}, 2));
        Parameter second = new Parameter(GradTensor.of(new float[] {3f, 4f, 5f, 6f}, 2, 2));
        Map<String, Parameter> named = new LinkedHashMap<>();
        named.put("weight", first);
        named.put("bias", second);

        assertEquals("weight:[2]:2;bias:[2, 2]:4;", TrainerMetadataSupport.parameterSignature(named));
        assertEquals("0:[2]:2;1:[2, 2]:4;", TrainerMetadataSupport.parameterSignature(List.of(first, second)));
    }
}
