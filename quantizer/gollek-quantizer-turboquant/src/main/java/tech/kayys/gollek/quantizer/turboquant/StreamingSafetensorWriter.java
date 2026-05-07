package tech.kayys.gollek.quantizer.turboquant;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.foreign.*;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/**
 * Streaming safetensor writer — writes large FP32 model files without
 * holding all dequantized tensors in memory simultaneously.
 *
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │ Previous approach (SafetensorConverter): │
 * │ Dequantize ALL layers → hold full float[] in heap → write once │
 * │ Memory peak = entire dequantized model (e.g., 70B FP32 = 280 GB) │
 * │ │
 * │ Streaming approach (this class): │
 * │ Pre-compute header from known shapes → write header stub │
 * │ For each layer: dequantize → stream write → free float[] │
 * │ Memory peak = ONE layer at a time (e.g., 4096×4096 = 64 MB) │
 * └────────────────────────────────────────────────────────────────────────┘
 *
 * Two-pass strategy (safetensor requires the header upfront):
 * Pass 1: Compute offsets from shapes (no data needed)
 * Pass 2: Write header + stream tensor data in order
 *
 * FFM write strategy:
 * Uses a sliding-window mmap over the output file. Each window covers
 * WRITE_WINDOW_SIZE bytes. When a tensor write crosses a window boundary,
 * the window is force-flushed and remapped forward. This avoids holding
 * the entire output file mapped in virtual address space.
 *
 * Usage:
 * 
 * <pre>{@code
 * var writer = new StreamingSafetensorWriter(outputPath, tensorPlan, metadata);
 * writer.open();
 * for (var layer : layers) {
 *     float[] w = dequantize(layer);
 *     writer.writeTensor(layer.getName() + ".weight", w);
 * }
 * writer.close();
 * }</pre>
 */
public class StreamingSafetensorWriter implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(StreamingSafetensorWriter.class);

    /** Size of sliding output mmap window (256 MB per window) */
    private static final long WRITE_WINDOW_SIZE = 256L * 1024 * 1024;

    private final Path outputPath;
    private final List<TensorPlan> tensorPlan; // ordered plan of tensors to write
    private final Map<String, String> metadata;
    private final ObjectMapper json = new ObjectMapper();

    // State after open()
    private FileChannel channel;
    private Arena arena;
    private MemorySegment window;
    private long windowBase; // file offset where current window starts
    private long writePos; // current absolute write position in file
    private long dataStart; // file offset where tensor data begins
    private byte[] headerBytes;
    private long totalFileSize;

    // Tensor plan entry
    public record TensorPlan(
            String name,
            long[] shape,
            long numElements // pre-computed product of shape dimensions
    ) {
        public long byteSize() {
            return numElements * Float.BYTES;
        }
    }

    public StreamingSafetensorWriter(
            Path outputPath,
            List<TensorPlan> tensorPlan,
            Map<String, String> metadata) {
        this.outputPath = outputPath;
        this.tensorPlan = tensorPlan;
        this.metadata = metadata != null ? metadata : Map.of();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Opens the output file and writes the JSON header.
     * Must be called before any {@link #writeTensor} calls.
     */
    public void open() throws IOException {
        // ── Pass 1: compute total size and build header JSON ──────────────────
        headerBytes = buildHeader();
        long jsonLen = headerBytes.length;
        dataStart = 8L + jsonLen;

        long dataSize = tensorPlan.stream().mapToLong(TensorPlan::byteSize).sum();
        totalFileSize = dataStart + dataSize;

        log.info("StreamingWriter: {} tensors, header={} bytes, data={:.2f} MB, total={:.2f} MB",
                tensorPlan.size(), jsonLen,
                dataSize / 1_048_576.0, totalFileSize / 1_048_576.0);

        // ── Allocate file ─────────────────────────────────────────────────────
        try (RandomAccessFile raf = new RandomAccessFile(outputPath.toFile(), "rw")) {
            raf.setLength(totalFileSize);
        }

        // ── Open channel and write header ─────────────────────────────────────
        channel = FileChannel.open(outputPath, StandardOpenOption.READ, StandardOpenOption.WRITE);
        arena = Arena.ofConfined();

        // Write 8-byte header length + JSON in one small mmap
        long headerRegion = dataStart;
        MemorySegment hdr = channel.map(FileChannel.MapMode.READ_WRITE, 0, headerRegion, arena);
        hdr.set(ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN), 0L, jsonLen);
        MemorySegment.copy(MemorySegment.ofArray(headerBytes), 0, hdr, 8, jsonLen);
        hdr.force();

        // ── Slide window to data region ───────────────────────────────────────
        writePos = dataStart;
        windowBase = dataStart;
        openWindow();

        log.debug("StreamingWriter opened: writing to {}", outputPath);
    }

    // ── Tensor Writing ────────────────────────────────────────────────────────

    /**
     * Writes one FP32 tensor to the output stream.
     *
     * The tensor must be in the same order it was declared in {@code tensorPlan}.
     * After writing, the float[] can be released for GC.
     *
     * @param name tensor name (must match an entry in tensorPlan)
     * @param data FP32 data array
     */
    public void writeTensor(String name, float[] data) throws IOException {
        long byteLen = (long) data.length * Float.BYTES;
        log.debug("Writing tensor '{}': {} elements ({} bytes)", name, data.length, byteLen);

        long remaining = byteLen;
        int srcOff = 0;

        while (remaining > 0) {
            long windowEnd = windowBase + WRITE_WINDOW_SIZE;
            long availInWindow = windowEnd - writePos;

            if (availInWindow <= 0) {
                // Slide window forward
                window.force();
                windowBase = writePos;
                openWindow();
                availInWindow = WRITE_WINDOW_SIZE;
            }

            long chunk = Math.min(remaining, availInWindow);
            long windowOff = writePos - windowBase;

            // Write chunk via FFM (JAVA_FLOAT, LE, element by element)
            int floatCount = (int) (chunk / Float.BYTES);
            for (int i = 0; i < floatCount; i++) {
                window.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN),
                        windowOff + (long) i * Float.BYTES,
                        data[srcOff + i]);
            }

            writePos += chunk;
            srcOff += floatCount;
            remaining -= chunk;
        }
    }

    /**
     * Writes a FP32 tensor from an off-heap MemorySegment (zero extra copy).
     *
     * @param name tensor name
     * @param src  source MemorySegment (FP32 LE, flat)
     */
    public void writeTensorFromSegment(String name, MemorySegment src) throws IOException {
        long byteLen = src.byteSize();
        long srcOff = 0L;

        while (srcOff < byteLen) {
            long windowEnd = windowBase + WRITE_WINDOW_SIZE;
            long availInWindow = windowEnd - writePos;

            if (availInWindow <= 0) {
                window.force();
                windowBase = writePos;
                openWindow();
                availInWindow = WRITE_WINDOW_SIZE;
            }

            long chunk = Math.min(byteLen - srcOff, availInWindow);
            long windowOff = writePos - windowBase;

            MemorySegment.copy(src, srcOff, window, windowOff, chunk);

            writePos += chunk;
            srcOff += chunk;
        }

        log.debug("Wrote tensor '{}' from segment: {} bytes", name, byteLen);
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    private void openWindow() throws IOException {
        long mapSize = Math.min(WRITE_WINDOW_SIZE, totalFileSize - windowBase);
        if (mapSize <= 0)
            return;
        window = channel.map(FileChannel.MapMode.READ_WRITE, windowBase, mapSize, arena);
    }

    private byte[] buildHeader() throws IOException {
        Map<String, Object> header = new LinkedHashMap<>();

        // Metadata block
        Map<String, String> meta = new LinkedHashMap<>(metadata);
        header.put("__metadata__", meta);

        // Compute running offsets
        long offset = 0L;
        for (TensorPlan plan : tensorPlan) {
            long byteSize = plan.byteSize();
            Map<String, Object> td = new LinkedHashMap<>();
            td.put("dtype", "F32");
            td.put("shape", Arrays.stream(plan.shape()).boxed().toList());
            td.put("data_offsets", List.of(offset, offset + byteSize));
            header.put(plan.name(), td);
            offset += byteSize;
        }

        return json.writeValueAsBytes(header);
    }

    /** Returns the current write position (useful for progress tracking). */
    public long getBytesWritten() {
        return writePos - dataStart;
    }

    /** Returns total data bytes to write. */
    public long getTotalDataBytes() {
        return totalFileSize - dataStart;
    }

    /** Progress as a fraction [0.0, 1.0]. */
    public double getProgress() {
        long total = getTotalDataBytes();
        return total == 0 ? 1.0 : (double) getBytesWritten() / total;
    }

    @Override
    public void close() throws IOException {
        if (window != null) {
            try {
                window.force();
            } catch (Exception ignored) {
            }
        }
        if (arena != null && arena.scope().isAlive()) {
            arena.close();
        }
        if (channel != null && channel.isOpen()) {
            channel.close();
        }
        log.info("StreamingWriter closed: wrote {:.2f} MB to {}",
                totalFileSize / 1_048_576.0, outputPath);
    }
}
