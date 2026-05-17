package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class DataLoaderTest {

    @Test
    void tensorLoaderBatchesRowsWithoutDroppingRemainder() {
        DataLoader.TensorDataLoader loader = DataLoader.tensors(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f}, 5, 1),
                2);

        List<DataLoader.Batch> batches = new ArrayList<>();
        for (DataLoader.Batch batch : loader) {
            batches.add(batch);
        }

        assertEquals(3, batches.size());
        assertEquals(5, loader.size());
        assertEquals(2, loader.batchSize());
        assertArrayEquals(new long[] {2, 1}, batches.get(0).inputs().shape());
        assertArrayEquals(new long[] {1, 1}, batches.get(2).inputs().shape());
        assertArrayEquals(new float[] {5f}, batches.get(2).inputs().data(), 1e-6f);
    }

    @Test
    void tensorLoaderSeedMakesShuffleDeterministic() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1));
        DataLoader.TensorDataLoader first = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(true)
                .seed(42L)
                .build();
        DataLoader.TensorDataLoader second = DataLoader.tensorBuilder(dataset)
                .batchSize(2)
                .shuffle(true)
                .seed(42L)
                .build();

        assertEquals(flattenInputs(first), flattenInputs(second));
    }

    @Test
    void tensorDatasetSplitIsDeterministicAndCoversAllSamples() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1),
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f}, 5, 1));

        DataLoader.TensorDatasetSplit first = dataset.split(0.6, 7L);
        DataLoader.TensorDatasetSplit second = dataset.split(0.6, 7L);

        assertEquals(3, first.train().size());
        assertEquals(2, first.validation().size());
        assertEquals(flattenDataset(first.train()), flattenDataset(second.train()));
        assertEquals(flattenDataset(first.validation()), flattenDataset(second.validation()));
    }

    @Test
    void tensorDatasetSplitBuildsReadyToTrainLoaders() {
        DataLoader.TensorDatasetSplit split = DataLoader.split(
                GradTensor.of(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, 6, 1),
                GradTensor.of(new float[] {2f, 4f, 6f, 8f, 10f, 12f}, 6, 1),
                0.67,
                11L);

        DataLoader.TensorDataLoader train = split.trainLoader(2, true, 99L);
        DataLoader.TensorDataLoader validation = split.validationLoader(2);

        assertEquals(4, train.size());
        assertEquals(2, validation.size());
        assertEquals(2, train.numBatches());
        assertEquals(1, validation.numBatches());
        assertEquals(flattenInputs(train), flattenInputs(split.trainLoader(2, true, 99L)));
    }

    @Test
    void classificationStratifiedSplitPreservesLabelsOnBothSides() {
        DataLoader.TensorDatasetSplit first = DataLoader.classificationStratifiedSplit(
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
        DataLoader.TensorDatasetSplit second = DataLoader.classificationStratifiedSplit(
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
    void binaryStratifiedSplitPreservesLabelsOnBothSides() {
        DataLoader.TensorDatasetSplit split = DataLoader.binaryStratifiedSplit(
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
    void positiveWeightHelpersDeriveBceImbalanceWeights() {
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
    void classWeightHelpersDeriveCrossEntropyImbalanceWeights() {
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
    void multiLabelBinaryStratifiedSplitBalancesPerLabelPositives() {
        DataLoader.TensorDatasetSplit split = DataLoader.multiLabelBinaryStratifiedSplit(
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
    void classificationLoaderBuildsClassIndexLabelBatches() {
        DataLoader.TensorDataLoader loader = DataLoader.classification(
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
        assertArrayEquals(new long[] {2, 2}, batches.get(0).inputs().shape());
        assertArrayEquals(new long[] {2}, batches.get(0).labels().shape());
        assertArrayEquals(new float[] {0f, 1f}, batches.get(0).labels().data(), 1e-6f);
        assertArrayEquals(new float[] {1f}, batches.get(1).labels().data(), 1e-6f);
    }

    @Test
    void binaryLoaderBuildsBceCompatibleColumnLabelBatches() {
        DataLoader.TensorDataLoader loader = DataLoader.binary(
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
    void binaryLoaderBuildsMultiLabelBceCompatibleLabelBatches() {
        DataLoader.TensorDataLoader loader = DataLoader.binary(
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
    void classificationDatasetRejectsInvalidLabels() {
        GradTensor inputs = GradTensor.of(new float[] {1f, 2f, 3f, 4f}, 2, 2);

        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classificationDataset(inputs, new int[] {0}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.classificationDataset(inputs, new int[] {0, -1}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[] {0}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[] {0, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[][] {{0, 1}, {1}}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[][] {{0, 2}, {1, 0}}));
        assertThrows(IllegalArgumentException.class,
                () -> DataLoader.binaryDataset(inputs, new int[][] {{0, 1}}));
    }

    @Test
    void tensorLoaderRejectsInvalidConfiguration() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f, 2f}, 2, 1));

        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorBuilder(dataset).batchSize(0));
        assertThrows(IllegalArgumentException.class, () -> dataset.split(1.0, 1L));
        assertThrows(IllegalArgumentException.class, () -> DataLoader.tensorDataset(
                GradTensor.of(new float[] {1f, 2f}, 2, 1),
                GradTensor.of(new float[] {1f}, 1, 1)));
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

    private static List<Float> flattenDataset(DataLoader.TensorDataset dataset) {
        List<Float> values = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
            for (float value : dataset.get(i)[0].data()) {
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
