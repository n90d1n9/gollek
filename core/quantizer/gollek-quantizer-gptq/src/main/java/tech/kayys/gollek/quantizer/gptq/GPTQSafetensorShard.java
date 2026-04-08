package tech.kayys.gollek.quantizer.gptq;

import tech.kayys.gollek.safetensor.loader.SafetensorDType;
import tech.kayys.gollek.safetensor.loader.SafetensorHeader;
import tech.kayys.gollek.safetensor.loader.SafetensorLoadResult;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorTensorInfo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.Set;

/**
 * GPTQ-specific view over a single safetensor shard, wrapping
 * {@link SafetensorLoadResult} from the safetensor-loader module.
 *
 * <p>Provides convenience methods for the tensor access patterns used by
 * the GPTQ quantizer:
 * <ul>
 *   <li>{@link #getTensorSegment(String)} — zero-copy MemorySegment for dequantization</li>
 *   <li>{@link #getTensorAsInt32(String)} — copy to int[] for qweight/qzeros</li>
 *   <li>{@link #getTensorAsFp16(String)} — copy to short[] for scales/bias</li>
 *   <li>{@link #getTensorShape(String)} — shape as List&lt;Long&gt;</li>
 *   <li>{@link #getTensorDtype(String)} — dtype string</li>
 * </ul>
 *
 * <p><b>Lifetime:</b> This object holds a reference to an open
 * {@link SafetensorLoadResult}. All tensor segments returned by this class
 * become invalid after {@link #close()} is called.
 */
public class GPTQSafetensorShard implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(GPTQSafetensorShard.class);

    private final Path shardPath;
    private final SafetensorLoadResult result;
    private volatile boolean closed = false;

    /**
     * Wraps an already-loaded {@link SafetensorLoadResult}.
     *
     * @param shardPath path to the shard file (for diagnostics)
     * @param result    the loaded result from SafetensorFFMLoader
     */
    public GPTQSafetensorShard(Path shardPath, SafetensorLoadResult result) {
        this.shardPath = shardPath;
        this.result = result;
    }

    // ── Header access ─────────────────────────────────────────────────────────

    /** Returns the parsed header for this shard. */
    public SafetensorHeader getHeader() {
        checkOpen();
        return result.header();
    }

    /** Returns file-level metadata (e.g., bits, group_size, desc_act). */
    public Map<String, String> getMetadata() {
        checkOpen();
        return result.header().fileMetadata();
    }

    /** Number of tensors in this shard. */
    public int tensorCount() {
        checkOpen();
        return result.tensorCount();
    }

    /** All tensor names in this shard. */
    public Set<String> tensorNames() {
        checkOpen();
        return result.tensorNames();
    }

    /** Whether this shard contains a tensor with the given name. */
    public boolean hasTensor(String name) {
        checkOpen();
        return result.header().hasTensor(name);
    }

    // ── Tensor access — MemorySegment (zero-copy) ─────────────────────────────

    /**
     * Returns a zero-copy MemorySegment for the named tensor.
     * The segment starts at byte 0 of the tensor's data and has
     * byteSize equal to the tensor's raw byte length.
     *
     * <p><b>WARNING:</b> the returned segment is only valid while this
     * shard is open. Do not use it after {@link #close()}.
     *
     * @param tensorName tensor key (e.g. "model.layers.0.self_attn.q_proj.qweight")
     * @return MemorySegment containing the tensor's raw bytes
     * @throws NoSuchElementException if the tensor is not in this shard
     */
    public MemorySegment getTensorSegment(String tensorName) {
        checkOpen();
        SafetensorTensor tensor = result.tensor(tensorName);
        return tensor.segment();
    }

    // ── Tensor access — typed arrays (copy to heap) ───────────────────────────

    /**
     * Copies the tensor data into an int[] array.
     * Valid only for I32 or U32 dtypes (e.g., qweight, qzeros, g_idx).
     */
    public int[] getTensorAsInt32(String tensorName) {
        checkOpen();
        SafetensorTensor tensor = result.tensor(tensorName);
        return tensor.toIntArray();
    }

    /**
     * Copies the tensor data into a short[] array (raw FP16 bits).
     * Valid only for F16, BF16, I16, or U16 dtypes (e.g., scales, bias).
     */
    public short[] getTensorAsFp16(String tensorName) {
        checkOpen();
        SafetensorTensor tensor = result.tensor(tensorName);
        return tensor.toShortArray();
    }

    /**
     * Copies the tensor data into a float[] array.
     * Valid only for F32 dtypes. For FP16 tensors, use {@link #getTensorAsFp16}
     * and convert via {@link MemoryAllocator#fp16ToFloat32}.
     */
    public float[] getTensorAsFloat32(String tensorName) {
        checkOpen();
        SafetensorTensor tensor = result.tensor(tensorName);
        return tensor.toFloatArray();
    }

    // ── Tensor metadata ───────────────────────────────────────────────────────

    /**
     * Returns the shape of a tensor as an unmodifiable list of dimensions.
     */
    public List<Long> getTensorShape(String tensorName) {
        checkOpen();
        SafetensorTensorInfo info = result.header().tensor(tensorName);
        long[] shape = info.shape();
        // Convert to List<Long> for compatibility with existing GPTQ code
        return Collections.unmodifiableList(
                java.util.stream.LongStream.of(shape).boxed().toList());
    }

    /**
     * Returns the dtype string for a tensor (e.g., "I32", "F16").
     */
    public String getTensorDtype(String tensorName) {
        checkOpen();
        SafetensorTensorInfo info = result.header().tensor(tensorName);
        return info.dtype().jsonName();
    }

    /**
     * Returns the raw byte size of a tensor's data.
     */
    public long getTensorByteSize(String tensorName) {
        checkOpen();
        SafetensorTensorInfo info = result.header().tensor(tensorName);
        return info.byteLength();
    }

    // ── FP16 convenience ──────────────────────────────────────────────────────

    /**
     * Reads a single FP16 element at the given flat index and converts it to float.
     * Valid only for F16 dtype tensors.
     */
    public float getF16ElementAsFloat(String tensorName, long flatIndex) {
        checkOpen();
        SafetensorTensor tensor = result.tensor(tensorName);
        return tensor.getF16AsFloat(flatIndex);
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Closes this shard, releasing the underlying memory-mapped file
     * or native allocation. All MemorySegments obtained from this shard
     * become invalid.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            result.close();
            log.debug("Closed GPTQ safetensor shard: {}", shardPath.getFileName());
        } catch (Exception e) {
            log.warn("Error closing shard {}: {}", shardPath.getFileName(), e.getMessage());
        }
    }

    public boolean isClosed() {
        return closed;
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "GPTQSafetensorShard is closed for shard: " + shardPath.getFileName());
        }
    }

    @Override
    public String toString() {
        return "GPTQSafetensorShard{path='%s', tensors=%d, closed=%b}"
                .formatted(shardPath.getFileName(), result.tensorCount(), closed);
    }
}
