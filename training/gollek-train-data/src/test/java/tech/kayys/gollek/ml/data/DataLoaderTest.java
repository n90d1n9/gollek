package tech.kayys.gollek.ml.data;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.ArrayList;
import java.util.Collections;
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

    @Test
    public void testTensorDatasetSplitAndSeededLoaderCompatibility() {
        var split = DataLoader.split(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f}, 6, 1),
                0.67,
                7L);

        var first = split.trainLoader(2, true, 42L);
        var second = split.trainLoader(2, true, 42L);

        assertEquals(4, split.train().size());
        assertEquals(2, split.validation().size());
        assertEquals(4, first.size());
        assertEquals(2, first.batchSize());
        assertTrue(first.shuffle());
        assertEquals(flattenInputs(first), flattenInputs(second));
        assertEquals(1, split.validationLoader(2).numBatches());
    }

    @Test
    public void testClassificationStratifiedSplitPreservesLabelsOnBothSides() {
        var first = DataLoader.classificationStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        1f, 1f,
                        0.2f, 0.8f,
                        0.8f, 0.2f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[] {0, 0, 1, 1, 0, 1, 0, 1},
                0.5,
                2026L);
        var second = DataLoader.classificationStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        1f, 1f,
                        0.2f, 0.8f,
                        0.8f, 0.2f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[] {0, 0, 1, 1, 0, 1, 0, 1},
                0.5,
                2026L);

        List<Float> trainLabels = flattenLabels(first.train());
        List<Float> validationLabels = flattenLabels(first.validation());
        assertEquals(4, first.train().size());
        assertEquals(4, first.validation().size());
        assertEquals(2, Collections.frequency(trainLabels, 0f));
        assertEquals(2, Collections.frequency(trainLabels, 1f));
        assertEquals(2, Collections.frequency(validationLabels, 0f));
        assertEquals(2, Collections.frequency(validationLabels, 1f));
        assertEquals(trainLabels, flattenLabels(second.train()));
        assertEquals(validationLabels, flattenLabels(second.validation()));
    }

    @Test
    public void testBinaryStratifiedSplitPreservesLabelsOnBothSides() {
        var split = DataLoader.binaryStratifiedSplit(
                GradTensor.of(new float[] {
                        -1f, -1f,
                        -0.8f, -0.7f,
                        -1.2f, -0.5f,
                        -0.6f, -1.1f,
                        1f, 1f,
                        0.9f, 0.7f,
                        1.2f, 0.5f,
                        0.6f, 1.1f
                }, 8, 2),
                new int[] {0, 0, 0, 0, 1, 1, 1, 1},
                0.5,
                99L);

        List<Float> trainLabels = flattenLabels(split.train());
        List<Float> validationLabels = flattenLabels(split.validation());
        assertEquals(4, split.train().size());
        assertEquals(4, split.validation().size());
        assertEquals(2, Collections.frequency(trainLabels, 0f));
        assertEquals(2, Collections.frequency(trainLabels, 1f));
        assertEquals(2, Collections.frequency(validationLabels, 0f));
        assertEquals(2, Collections.frequency(validationLabels, 1f));
    }

    @Test
    public void testPositiveWeightHelpersDeriveBceImbalanceWeights() {
        assertEquals(3.0f, DataLoader.binaryPositiveWeight(1, 0, 0, 0), 1e-6f);
        assertEquals(1.0f, DataLoader.binaryPositiveWeight(true, false), 1e-6f);
        assertArrayEquals(new float[] {3.0f, 3.0f, 1.0f}, DataLoader.multiLabelPositiveWeights(new int[][] {
                {1, 0, 1},
                {0, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
        }), 1e-6f);
    }

    @Test
    public void testClassWeightHelpersDeriveCrossEntropyImbalanceWeights() {
        assertArrayEquals(
                new float[] {2.0f / 3.0f, 2.0f},
                DataLoader.classWeights(0, 0, 0, 1),
                1e-6f);
        assertArrayEquals(
                new float[] {4.0f / 9.0f, 4.0f / 3.0f, 1.0f},
                DataLoader.classWeightsFor(3, 0, 0, 0, 1),
                1e-6f);
        assertThrows(IllegalArgumentException.class, () -> DataLoader.classWeightsFor(2, 0, 2));
    }

    @Test
    public void testMultiLabelBinaryStratifiedSplitBalancesPerLabelPositives() {
        var split = DataLoader.multiLabelBinaryStratifiedSplit(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0.9f, 0.1f,
                        0f, 1f,
                        0.1f, 0.9f,
                        0.5f, 0.5f,
                        0.6f, 0.4f,
                        0.2f, 0.8f,
                        0.3f, 0.7f
                }, 8, 2),
                new int[][] {
                        {1, 0, 0},
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                        {0, 0, 1},
                        {1, 1, 1},
                        {1, 1, 1}
                },
                0.5,
                123L);

        assertEquals(4, split.train().size());
        assertEquals(4, split.validation().size());
        assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(split.train(), 3));
        assertArrayEquals(new int[] {2, 2, 2}, positiveCounts(split.validation(), 3));
    }

    @Test
    public void testClassificationLoaderBuildsClassIndexLabels() {
        var loader = DataLoader.classification(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0f, 1f,
                        1f, 1f
                }, 3, 2),
                new int[] {0, 1, 1},
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {0f, 1f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new float[] {1f}, batches.get(1).labels().data(), 1e-6f);
    }

    @Test
    public void testBinaryLoaderBuildsBceCompatibleColumnLabels() {
        var loader = DataLoader.binary(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0f, 1f,
                        1f, 1f
                }, 3, 2),
                new int[] {1, 0, 1},
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2, 1}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {1f, 0f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {1, 1}, batches.get(1).labels().shape());
        assertArrayEquals(new float[] {1f}, batches.get(1).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {3, 1}, DataLoader.binaryLabels(true, false, true).shape());
    }

    @Test
    public void testBinaryLoaderBuildsMultiLabelBceCompatibleLabels() {
        var loader = DataLoader.binary(
                GradTensor.of(new float[] {
                        1f, 0f,
                        0f, 1f,
                        1f, 1f
                }, 3, 2),
                new int[][] {
                        {1, 0, 1},
                        {0, 1, 0},
                        {1, 1, 0}
                },
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(2, batches.size());
        assertArrayEquals(new long[] {2, 3}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {1f, 0f, 1f, 0f, 1f, 0f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {1, 3}, batches.get(1).labels().shape());
        assertArrayEquals(new float[] {1f, 1f, 0f}, batches.get(1).labels().data(), 1e-6f);
        assertArrayEquals(new long[] {2, 2}, DataLoader.binaryLabels(new boolean[][] {
                {true, false},
                {false, true}
        }).shape());
    }

    @Test
    public void testTensorLoaderRejectsInvalidConfigEarly() {
        var dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {2f, 4f}, 2, 1));

        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorBuilder(dataset).batchSize(0));
        assertThrows(IllegalArgumentException.class, () -> dataset.split(0.0, 1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f}, 1, 1)));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.classificationDataset(
                GradTensor.of(new float[] {1f, 2f}, 1, 2),
                new int[] {0, -1}));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.binaryDataset(
                GradTensor.of(new float[] {1f, 2f}, 1, 2),
                new int[] {2}));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.binaryDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2),
                new int[][] {{0, 1}, {1}}));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.binaryDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2),
                new int[][] {{0, 2}, {1, 0}}));
    }

    private static List<Float> flattenInputs(DataLoader.TensorDataLoader loader) {
        List<Float> values = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            for (float value : batch.inputs().data()) {
                values.add(value);
            }
        }
        return values;
    }

    private static List<Float> flattenLabels(DataLoader.TensorDataset dataset) {
        List<Float> values = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            for (float value : dataset.get(i)[1].data()) {
                values.add(value);
            }
        }
        return values;
    }

    private static int[] positiveCounts(DataLoader.TensorDataset dataset, int columns) {
        int[] counts = new int[columns];
        for (int i = 0; i < dataset.size(); i++) {
            float[] labels = dataset.get(i)[1].data();
            assertEquals(columns, labels.length);
            for (int column = 0; column < columns; column++) {
                if (labels[column] >= 0.5f) {
                    counts[column]++;
                }
            }
        }
        return counts;
    }
}
