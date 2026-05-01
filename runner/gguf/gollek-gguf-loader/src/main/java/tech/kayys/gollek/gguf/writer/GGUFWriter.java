package tech.kayys.gollek.gguf.writer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Writer for GGUF format files.
 */
public final class GGUFWriter {
    
    private final ByteBuffer buffer;
    private final List<TensorInfoEntry> tensorInfos = new ArrayList<>();
    
    private static class TensorInfoEntry {
        String name;
        long[] shape;
        int dtype;
        long offset;
        
        TensorInfoEntry(String name, long[] shape, int dtype, long offset) {
            this.name = name;
            this.shape = shape;
            this.dtype = dtype;
            this.offset = offset;
        }
    }
    
    public GGUFWriter() {
        this.buffer = ByteBuffer.allocate(1024 * 1024);  // Start with 1MB
        this.buffer.order(ByteOrder.LITTLE_ENDIAN);
    }
    
    public void writeHeader(int version) {
        // Magic "GGUF"
        writeBytes("GGUF".getBytes(StandardCharsets.UTF_8));
        writeInt32(version);
        // Placeholder for tensor count and KV count (will be updated later)
        writeInt64(0);  // Tensor count placeholder
        writeInt64(0);  // KV count placeholder
    }
    
    public void writeString(String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        writeInt64(bytes.length);
        writeBytes(bytes);
    }
    
    public void writeInt32(int value) {
        ensureCapacity(4);
        buffer.putInt(value);
    }
    
    public void writeInt64(long value) {
        ensureCapacity(8);
        buffer.putLong(value);
    }
    
    public void writeFloat32(float value) {
        ensureCapacity(4);
        buffer.putFloat(value);
    }
    
    public void writeBool(boolean value) {
        ensureCapacity(1);
        buffer.put((byte) (value ? 1 : 0));
    }
    
    public void writeBytes(byte[] bytes) {
        ensureCapacity(bytes.length);
        buffer.put(bytes);
    }
    
    public void writeBytes(ByteBuffer src) {
        ensureCapacity(src.remaining());
        buffer.put(src);
    }
    
    public void writeTensorInfo(String name, long[] shape, int dtype, long offset) {
        tensorInfos.add(new TensorInfoEntry(name, shape, dtype, offset));
    }
    
    public void alignTo(int alignment) {
        long pos = buffer.position();
        long padding = (alignment - (pos % alignment)) % alignment;
        if (padding > 0) {
            ensureCapacity((int) padding);
            for (int i = 0; i < padding; i++) {
                buffer.put((byte) 0);
            }
        }
    }
    
    public long position() {
        return buffer.position();
    }
    
    public void updateTensorOffsets(long dataStart) {
        // This would update the tensor info section with actual offsets
        // In production, we'd need to rewrite the header
    }
    
    public byte[] toByteArray() {
        byte[] result = new byte[buffer.position()];
        buffer.rewind();
        buffer.get(result);
        return result;
    }
    
    private void ensureCapacity(int needed) {
        if (buffer.remaining() < needed) {
            ByteBuffer newBuffer = ByteBuffer.allocate(buffer.capacity() * 2);
            newBuffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.flip();
            newBuffer.put(buffer);
            this.buffer.position(newBuffer.position());
        }
    }
}