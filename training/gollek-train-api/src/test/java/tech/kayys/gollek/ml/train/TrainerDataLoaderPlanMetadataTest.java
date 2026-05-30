package tech.kayys.gollek.ml.train;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.train.data.DataLoader;
import tech.kayys.gollek.train.data.DataLoader.Batch;

class TrainerDataLoaderPlanMetadataTest {

    @Test
    void capturesTensorLoaderPlansWithoutConsumingLoaderEpochs() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 4, 1),
                GradTensor.of(new float[] {0f, 1f, 0f, 1f}, 4, 1));
        DataLoader.TensorDataLoader train = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(99L)
                .reshuffleEachEpoch()
                .build();

        Map<String, Object> metadata = TrainerDataLoaderPlanMetadata.capture(train, null);

        assertEquals(Boolean.TRUE, metadata.get("dataLoaderPlanMetadataCaptured"));
        assertEquals(Boolean.TRUE, metadata.get("trainLoaderPlan.available"));
        assertEquals("tensor", metadata.get("trainLoaderPlan.kind"));
        assertEquals(4, metadata.get("trainLoaderPlan.datasetSize"));
        assertEquals(4, metadata.get("trainLoaderPlan.sampleCount"));
        assertEquals(2, metadata.get("trainLoaderPlan.batchSize"));
        assertEquals(2, metadata.get("trainLoaderPlan.batchCount"));
        assertEquals(Boolean.TRUE, metadata.get("trainLoaderPlan.shuffle"));
        assertEquals(99L, metadata.get("trainLoaderPlan.shuffleSeed"));
        assertEquals(Boolean.TRUE, metadata.get("trainLoaderPlan.reshuffleEachEpoch"));
        assertEquals(Boolean.FALSE, metadata.get("trainLoaderPlan.derivedFromBatchCollection"));
        assertEquals(Boolean.FALSE, metadata.get("validationLoaderPlan.available"));
        assertEquals("no-loader", metadata.get("validationLoaderPlan.skipReason"));
        assertEquals(0L, train.initialEpoch());
    }

    @Test
    void derivesPlansForMaterializedBatchCollections() {
        List<Batch> train = List.of(
                batch(new float[] {1f, 2f}, new float[] {2f, 4f}, 2),
                batch(new float[] {3f}, new float[] {6f}, 1));

        Map<String, Object> metadata = TrainerDataLoaderPlanMetadata.capture(train, List.of());

        assertEquals(Boolean.TRUE, metadata.get("trainLoaderPlan.available"));
        assertEquals("batch-collection", metadata.get("trainLoaderPlan.kind"));
        assertEquals(3, metadata.get("trainLoaderPlan.datasetSize"));
        assertEquals(3, metadata.get("trainLoaderPlan.sampleCount"));
        assertEquals(2, metadata.get("trainLoaderPlan.batchSize"));
        assertEquals(2, metadata.get("trainLoaderPlan.batchCount"));
        assertEquals(Boolean.TRUE, metadata.get("trainLoaderPlan.derivedFromBatchCollection"));
        assertEquals(Boolean.FALSE, metadata.get("validationLoaderPlan.available"));
        assertEquals("empty-batch-collection", metadata.get("validationLoaderPlan.skipReason"));
    }

    private static Batch batch(float[] inputs, float[] labels, int rows) {
        return new Batch(
                GradTensor.of(inputs, rows, 1),
                GradTensor.of(labels, rows, 1));
    }
}
