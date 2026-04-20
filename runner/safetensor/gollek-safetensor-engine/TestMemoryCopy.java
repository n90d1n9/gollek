import java.lang.foreign.*;
import java.nio.ByteOrder;
public class TestMemoryCopy {
    public static void main(String[] args) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment src = arena.allocate(100 * 4);
            MemorySegment dst = arena.allocate(100 * 4);
            for(int i=0; i<100; i++) {
                src.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)i);
            }
            // Try copying element 10 to dst at offset 0.
            // If srcOffset is BYTES, passing 10 should copy from byte 10.
            // If srcOffset is ELEMENTS, passing 10 should copy from element 10 (byte 40).
            MemorySegment.copy(src, ValueLayout.JAVA_FLOAT, 10, dst, ValueLayout.JAVA_FLOAT, 0, 1);
            System.out.println("dst[0] = " + dst.getAtIndex(ValueLayout.JAVA_FLOAT, 0));
        }
    }
}
