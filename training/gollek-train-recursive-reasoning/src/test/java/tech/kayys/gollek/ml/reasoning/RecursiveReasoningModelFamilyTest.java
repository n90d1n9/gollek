package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class RecursiveReasoningModelFamilyTest {
    @Test
    void exposesExpectedMetadata() {
        assertEquals("generative-recursive-reasoning", RecursiveReasoningModelFamily.FAMILY_ID);
        assertEquals("GRAM", RecursiveReasoningModelFamily.SHORT_NAME);
        assertTrue(RecursiveReasoningModelFamily.PAPER_CITATION.contains("2605.19376"));
    }

    @Test
    void recommendsRecursiveReasoningModules() {
        List<String> modules = RecursiveReasoningModelFamily.recommendedModuleIds();
        assertTrue(modules.contains("ml:gollek-ml-recursive-reasoning"));
        assertTrue(modules.contains("trainer:gollek-trainer-recursive-reasoning"));
    }
}
