package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.quantizer.turboquant.BnBDequantizer;
import org.jboss.logging.Logger;

/**
 * Adapter for BitsAndBytes (BnB) quantization formats (NF4, INT8).
 */
public class BnBQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(BnBQuantizerAdapter.class);
    private final BnBDequantizer engine = new BnBDequantizer();

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        // BnB is primarily used for dequantization of pre-quantized models.
        // Implementing the quantization (compression) side if needed.
        throw new UnsupportedOperationException("BnB quantization side not implemented yet");
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        log.debugf("BnB: dequantizing NF4/INT8 tensor %s", java.util.Arrays.toString(quantizedTensor.shape()));
        
        // Extract raw data and metadata from AccelTensor
        java.lang.foreign.MemorySegment dataSeg = quantizedTensor.dataSegment();
        java.lang.foreign.MemorySegment scalesSeg = quantizedTensor.scales();
        
        if (scalesSeg == null) {
            log.warn("BnB dequantization requested but no scales found on tensor");
            return quantizedTensor;
        }

        int numElements = (int) quantizedTensor.numel();
        float[] output = new float[numElements];
        
        // Map MemorySegment to byte array (copy for now, optimized engine can use segments)
        byte[] packedBytes = dataSeg.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
        float[] absmax = scalesSeg.toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);

        // Run the optimized SIMD dequantization from core
        engine.dequantNF4(packedBytes, absmax, numElements, output);
        
        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    @Override
    public String getName() {
        return "BitsAndBytes";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getStrategy() == tech.kayys.gollek.safetensor.quantization.QuantizationEngine.QuantStrategy.INT8 
            || config.getBits() == 4; // NF4
    }
}
