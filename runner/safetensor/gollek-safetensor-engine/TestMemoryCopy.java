import java.lang.foreign.*;
import java.nio.ByteOrder;

/**
 * Verification of MemorySegment.copy behavior in JDK 25.
 * Confirms that passing byte offsets (scaled by element size) is correct.
 */
public class TestMemoryCopy {
    public static void main(String[] args) {
        try (Arena arena = Arena.ofConfined()) {
            int count = 100;
            MemorySegment src = arena.allocate(count * 4);
            MemorySegment dst = arena.allocate(count * 4);
            
            // Fill src with sequential floats
            for(int i = 0; i < count; i++) {
                src.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float)i);
            }
            
            // Test: Copy element 10 to dst index 0.
            // ELEMENT 10 is at BYTE OFFSET 40.
            long srcByteOffset = 10 * ValueLayout.JAVA_FLOAT.byteSize();
            long dstByteOffset = 0;
            long elementCount = 1;
            
            MemorySegment.copy(src, ValueLayout.JAVA_FLOAT, srcByteOffset, 
                               dst, ValueLayout.JAVA_FLOAT, dstByteOffset, 
                               elementCount);
            
            float result = dst.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
            System.out.println("Copying from element 10 (byte offset " + srcByteOffset + ")");
            System.out.println("Result dst[0]: " + result);
            
            if (result == 10.0f) {
                System.out.println("SUCCESS: Byte-based offset confirmed.");
            } else {
                System.out.println("FAILURE: Incorrect value copied.");
            }
        }
    }
}
