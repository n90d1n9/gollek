package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class ByteLatentModelFamilyTest {

    @Test
    void exposesCitationAndRecommendedModules() {
        assertEquals("fast-byte-latent-transformer", ByteLatentModelFamily.FAMILY_ID);
        assertTrue(ByteLatentModelFamily.PAPER_CITATION.contains("arXiv:2605.08044v1"));

        List<String> modules = ByteLatentModelFamily.recommendedModuleIds();
        assertTrue(modules.contains("ml:gollek-ml-byte-latent"));
        assertTrue(modules.contains("trainer:gollek-trainer-byte-latent"));
    }
}
