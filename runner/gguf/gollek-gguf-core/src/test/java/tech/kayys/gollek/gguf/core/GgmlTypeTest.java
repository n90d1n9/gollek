package tech.kayys.gollek.gguf.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GgmlTypeTest {
    @Test
    void q8_1UsesGgmlReferenceBlockLayout() {
        assertEquals(32, GgmlType.Q8_1.blockSize);
        assertEquals(36, GgmlType.Q8_1.typeSize);
        assertEquals(36L, GgmlType.Q8_1.bytesFor(32));
        assertEquals(72L, GgmlType.Q8_1.bytesFor(64));
        assertEquals(36L, GGUFConstants.tensorBytes(GGUFConstants.GGML_TYPE_Q8_1, 32));
    }
}
