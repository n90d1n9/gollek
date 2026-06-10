package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;

class TensorBatchCollatorTest {
    @Test
    void collatePassesSelectedIndicesToCollateFunction() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {10f, 20f, 30f}, 3, 1),
                GradTensor.of(new float[] {1f, 2f, 3f}, 3, 1));

        DataLoader.Batch batch = TensorBatchCollator.collate(
                DataLoader.defaultTensorCollate(),
                List.of(2, 0),
                dataset);

        assertArrayEquals(new float[] {30f, 10f}, batch.inputs().data(), 1e-6f);
        assertArrayEquals(new float[] {3f, 1f}, batch.labels().data(), 1e-6f);
    }

    @Test
    void collateRejectsNullInputs() {
        DataLoader.TensorDataset dataset = DataLoader.tensorDataset(
                GradTensor.of(new float[] {10f}, 1, 1),
                GradTensor.of(new float[] {1f}, 1, 1));

        assertThrows(NullPointerException.class, () -> TensorBatchCollator.collate(null, List.of(0), dataset));
        assertThrows(
                NullPointerException.class,
                () -> TensorBatchCollator.collate(DataLoader.defaultTensorCollate(), null, dataset));
        assertThrows(
                NullPointerException.class,
                () -> TensorBatchCollator.collate(DataLoader.defaultTensorCollate(), List.of(0), null));
    }
}
