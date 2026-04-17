package tech.kayys.gollek.tensorflow;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.tensorflow.loader.TFGraphParser;
import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Path;

class TFGraphParserTest {
    @Test
    void parseStubReturnsModel() {
        TFGraphParser p = new TFGraphParser();
        assertNotNull(p.parseFrozenPb(Path.of("dummy.pb")));
    }
}
