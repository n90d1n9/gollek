package tech.kayys.gollek.onnx;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.onnx.loader.ONNXParser;
import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Path;

class ONNXParserTest {
    @Test
    void parseStubReturnsModel() {
        ONNXParser p = new ONNXParser();
        assertNotNull(p.parse(Path.of("dummy.onnx")));
    }
}
