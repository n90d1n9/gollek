package tech.kayys.gollek.ml.data;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class DataLoaderTest {

    @Test
    public void testDataLoaderBatching() {
        var inputs = GradTensor.randn(10, 4);
        var targets = GradTensor.randn(10, 2);
        var dataset = new DataLoader.TensorDataset(inputs, targets);
        var loader = DataLoader.tensorBuilder(dataset)
            .batchSize(3)
            .shuffle(false)
            .dropLast(false)
            .build();

        int batchCount = 0;
        int totalSamples = 0;
        for (DataLoader.Batch batch : loader) {
            int bs = (int) batch.inputs().shape()[0];
            assertTrue(bs <= 3);
            totalSamples += bs;
            batchCount++;
        }
        assertEquals(4, batchCount); // ceil(10/3) = 4
        assertEquals(10, totalSamples);
    }

    @Test
    public void testDataLoaderDropLast() {
        var inputs = GradTensor.randn(10, 4);
        var targets = GradTensor.randn(10, 2);
        var dataset = new DataLoader.TensorDataset(inputs, targets);
        var loader = DataLoader.tensorBuilder(dataset)
            .batchSize(3)
            .shuffle(false)
            .dropLast(true)
            .build();

        int batchCount = 0;
        for (DataLoader.Batch batch : loader) {
            assertEquals(3, batch.inputs().shape()[0]);
            batchCount++;
        }
        assertEquals(3, batchCount); // 10 / 3 = 3 full batches
    }

    @Test
    public void testDataLoaderNumBatches() {
        var inputs = GradTensor.randn(20, 4);
        var targets = GradTensor.randn(20, 2);
        var dataset = new DataLoader.TensorDataset(inputs, targets);

        var loader1 = DataLoader.tensorBuilder(dataset).batchSize(5).build();
        assertEquals(4, loader1.numBatches());

        var loader2 = DataLoader.tensorBuilder(dataset).batchSize(7).build();
        assertEquals(3, loader2.numBatches());

        var loader3 = DataLoader.tensorBuilder(dataset).batchSize(7).dropLast(true).build();
        assertEquals(2, loader3.numBatches());
    }
}
