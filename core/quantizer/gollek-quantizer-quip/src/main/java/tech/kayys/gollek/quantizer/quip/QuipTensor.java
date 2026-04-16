package tech.kayys.gollek.quantizer.quip;

/**
 * A QuIP#-quantized weight tensor.
 *
 * <p>Stores the compressed representation of one weight matrix W of shape
 * [rows × cols] after incoherence processing and E8 vector quantization.
 *
 * <p>Layout:
 * <ul>
 *   <li>{@code codes}  — byte[] of length (rows*cols/8), one byte per 8-dim block</li>
 *   <li>{@code scales} — float[] of length (rows*cols/8), one scale per block</li>
 *   <li>{@code seedU, seedV} — seeds to reconstruct the Hadamard transforms</li>
 *   <li>{@code rows, cols} — original matrix dimensions</li>
 * </ul>
 */
public record QuipTensor(
        String name,
        int rows,
        int cols,
        byte[] codes,    // E8 codeword index per 8-dim block (one byte = index 0-255)
        float[] scales,  // per-block scale factor
        long seedU,
        long seedV
) {
    /** Number of 8-dim blocks. */
    public int numBlocks() { return codes.length; }

    /** Compressed size in bytes. */
    public long compressedBytes() { return codes.length + (long) scales.length * Float.BYTES; }

    /** Original size in bytes (float32). */
    public long originalBytes() { return (long) rows * cols * Float.BYTES; }

    /** Compression ratio. */
    public double compressionRatio() { return (double) originalBytes() / compressedBytes(); }
}
