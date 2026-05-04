package tech.kayys.gollek.ml.nlp;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.tensor.Tensor;

import static org.junit.jupiter.api.Assertions.*;

public class NLPFeatureTest {

    @Test
    public void testBasicPipeline() {
        Language nlp = LanguageFactory.load("en_core_web_sm");
        Doc doc = nlp.process("Gollek is a powerful ML framework");

        assertEquals(6, doc.length());
        assertEquals("Gollek", doc.get(0).getText());
        assertEquals("framework", doc.get(5).getText());
        assertTrue(doc.get(0).isAlpha());
    }

    @Test
    public void testVectorSimilarity() {
        // Mock vectors for testing
        Tensor v1 = Tensor.of(1.0f, 0.0f, 0.0f);
        Tensor v2 = Tensor.of(0.0f, 1.0f, 0.0f);
        Tensor v3 = Tensor.of(1.0f, 0.1f, 0.0f);

        Doc doc = new Doc("test");
        doc.setVector(v1);

        Doc other1 = new Doc("other1");
        other1.setVector(v2);

        Doc other2 = new Doc("other2");
        other2.setVector(v3);

        double sim1 = doc.similarity(other1);
        double sim2 = doc.similarity(other2);

        assertTrue(sim1 < 0.001); // Near zero (orthogonal)
        assertTrue(sim2 > 0.9); // Near one (similar)
    }
}
