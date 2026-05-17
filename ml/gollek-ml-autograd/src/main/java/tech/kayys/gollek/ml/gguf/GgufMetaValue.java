package tech.kayys.gollek.ml.gguf;

import java.io.Serial;
import java.io.Serializable;

/**
 * Temporary GGUF metadata compatibility surface for the ML/training side.
 */
public sealed interface GgufMetaValue extends Serializable permits GgufMetaValue.StringVal, GgufMetaValue.Uint32Val {

    @Serial
    long serialVersionUID = 1L;

    record StringVal(String value) implements GgufMetaValue {
        @Serial
        private static final long serialVersionUID = 1L;
    }

    record Uint32Val(long value) implements GgufMetaValue {
        @Serial
        private static final long serialVersionUID = 1L;
    }

    static GgufMetaValue ofString(String value) {
        return new StringVal(value);
    }

    static GgufMetaValue ofUint32(long value) {
        return new Uint32Val(value);
    }
}
