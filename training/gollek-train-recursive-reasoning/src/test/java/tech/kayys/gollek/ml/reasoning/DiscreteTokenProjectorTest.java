package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class DiscreteTokenProjectorTest {
    @Test
    void projectsItemLogitsWithArgmax() {
        float[] logits = {
            -1.0f, 2.0f, 0.0f,
            4.0f, 1.0f, 2.0f,
            -1.0f, 0.0f, 3.0f
        };

        DiscreteTokenProjectionResult result = DiscreteTokenProjector.argmax(3, 3, logits);

        assertArrayEquals(new int[] {1, 0, 2}, result.tokens());
        assertEquals(3, result.itemCount());
        assertEquals(3, result.vocabSize());
        assertEquals("argmax", result.metadata().get("projection"));
        result.tokens()[0] = 9;
        assertArrayEquals(new int[] {1, 0, 2}, result.tokens());
    }

    @Test
    void findsBestItemForOneTokenInsideRange() {
        float[] logits = {
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 5.0f,
            0.0f, 0.0f, 3.0f,
            0.0f, 0.0f, 9.0f
        };

        assertEquals(1, DiscreteTokenProjector.bestItemForToken(logits, 0, 3, 3, 2));
        assertEquals(3, DiscreteTokenProjector.bestItemForToken(logits, 2, 2, 3, 2));
        assertEquals(2, DiscreteTokenProjector.argmaxTokenAt(logits, 0, 3));
    }

    @Test
    void rejectsInvalidLogitLayouts() {
        assertThrows(IllegalArgumentException.class, () -> DiscreteTokenProjector.argmax(0, 3, new float[] {}));
        assertThrows(IllegalArgumentException.class, () -> DiscreteTokenProjector.argmax(2, 0, new float[] {}));
        assertThrows(IllegalArgumentException.class, () -> DiscreteTokenProjector.argmax(2, 3, new float[5]));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenProjector.argmaxTokenAt(new float[5], 0, 3));
        assertThrows(
                IllegalArgumentException.class,
                () -> DiscreteTokenProjector.bestItemForToken(new float[6], 0, 1, 3, 3));

        float[] logits = new float[6];
        logits[4] = Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, () -> DiscreteTokenProjector.argmax(2, 3, logits));
    }
}
