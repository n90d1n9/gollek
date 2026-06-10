package tech.kayys.gollek.train.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class GenericBatchMaterializerTest {
    @Test
    void materializePreservesIndexOrderAndDuplicates() {
        Dataset<String> dataset = Dataset.of("zero", "one", "two", "three");

        assertEquals(
                List.of("three", "one", "three", "zero"),
                GenericBatchMaterializer.materialize(dataset, List.of(3, 1, 3, 0)));
    }

    @Test
    void materializeRejectsNullInputs() {
        Dataset<String> dataset = Dataset.of("zero");

        assertThrows(NullPointerException.class, () -> GenericBatchMaterializer.materialize(null, List.of(0)));
        assertThrows(NullPointerException.class, () -> GenericBatchMaterializer.materialize(dataset, null));
    }

    @Test
    void materializePropagatesInvalidDatasetIndex() {
        Dataset<String> dataset = Dataset.of("zero");

        assertThrows(IndexOutOfBoundsException.class, () -> GenericBatchMaterializer.materialize(dataset, List.of(1)));
    }
}
