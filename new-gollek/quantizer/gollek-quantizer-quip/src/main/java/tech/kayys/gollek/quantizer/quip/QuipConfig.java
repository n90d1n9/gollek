package tech.kayys.gollek.quantizer.quip;

/**
 * QuIP# quantization configuration.
 *
 * @param bits        bits per weight (2 = E8 codebook, 4 = scalar fallback)
 * @param hadamardSeed seed for the random Hadamard transforms U and V
 * @param scalarFallback if true, use scalar INT4 for layers whose dimensions
 *                       are not divisible by {@link E8Codebook#DIM}
 */
public record QuipConfig(int bits, long hadamardSeedU, long hadamardSeedV, boolean scalarFallback) {

    public static QuipConfig quip2bit() {
        return new QuipConfig(2, 0xDEADBEEFL, 0xCAFEBABEL, true);
    }

    public static QuipConfig quip4bit() {
        return new QuipConfig(4, 0xDEADBEEFL, 0xCAFEBABEL, true);
    }
}
