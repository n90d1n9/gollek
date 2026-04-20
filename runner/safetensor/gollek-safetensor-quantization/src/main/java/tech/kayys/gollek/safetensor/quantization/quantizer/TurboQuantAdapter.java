package tech.kayys.gollek.safetensor.quantization.quantizer;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.quantization.QuantConfig;
import tech.kayys.gollek.quantizer.turboquant.TurboQuantEngine;
import tech.kayys.gollek.quantizer.turboquant.TurboQuantConfig;
import org.jboss.logging.Logger;

import java.util.Arrays;

/**
 * Adapter that bridges the optimized TurboQuant core engine with the Safetensor runner.
 */
public class TurboQuantAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(TurboQuantAdapter.class);

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        if (tensor == null) throw new IllegalArgumentException("Tensor cannot be null");
        
        log.infof("TurboQuant: quantizing tensor %s (bits=%d)", 
                 Arrays.toString(tensor.shape()), config.getBits());

        // Map runner QuantConfig to TurboQuantConfig
        int dim = (int) (tensor.numel());
        TurboQuantConfig tqConfig = new TurboQuantConfig(
            config.getBits(),
            dim,
            TurboQuantConfig.Variant.MSE,
            TurboQuantConfig.RotationStrategy.HADAMARD,
            false, // splitOutliers
            0,     // outlierChannels
            42L    // seed
        );

        TurboQuantEngine engine = new TurboQuantEngine(tqConfig);
        float[] data = tensor.toFloatArray();
        
        // TurboQuant_prod for unbiased inner product estimation
        TurboQuantEngine.QuantProdResult result = engine.quantizeProd(data);
        
        // TODO: Map TurboQuant Engine results back to AccelTensor bit-layout
        // This requires implementing the bit-packing for QuantProdResult in AccelTensor.
        log.warn("TurboQuant bit-packing to AccelTensor not yet implemented in adapter");
        
        return tensor; // Placeholder
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        log.infof("TurboQuant: dequantizing tensor");
        // Implementation will depend on how we store TurboQuant results in AccelTensor
        return quantizedTensor;
    }

    @Override
    public String getName() {
        return "TurboQuant";
    }

    @Override
    public boolean supports(QuantConfig config) {
        // TurboQuant specialized in 2-4 bit vector quantization
        return config.getBits() <= 4;
    }
}
