package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.gguf.loader.GGUFModel;
import tech.kayys.gollek.gguf.loader.GGUFLoader;
import tech.kayys.gollek.gguf.loader.GGUFTensorInfo;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.tensor.weights.TransformerLayerWeights;
import tech.kayys.gollek.spi.model.ModelArchitecture;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Adapter to bridge GGUF loaded weights to AccelTensor format used by the Java-native engine.
 */
public class GgufWeightAdapter {

    public static Map<String, AccelTensor> adaptWeights(GGUFModel model, List<TransformerLayerWeights> layers, ModelArchitecture arch) {
        Map<String, AccelTensor> weights = new HashMap<>();

        // Map non-layer weights (embeddings, final norm, lm_head)
        weights.put(arch.embedTokensWeight(), wrap(model, "token_embd.weight"));
        weights.put(arch.finalNormWeight(), wrap(model, "output_norm.weight"));
        
        // Handle lm_head (output.weight in GGUF, often tied to embeddings)
        String lmHeadName = "output.weight";
        if (model.tensors().stream().anyMatch(t -> t.name().equals(lmHeadName))) {
            weights.put(arch.lmHeadWeight(), wrap(model, lmHeadName));
        }

        // Map layer weights
        for (int i = 0; i < layers.size(); i++) {
            TransformerLayerWeights lw = layers.get(i);
            weights.put(arch.layerAttentionNormWeight(i), AccelTensor.wrapSegment(lw.rmsWeight, new long[]{(long)lw.rmsWeight.byteSize() / 4}));
            
            // Note: GGUF runner prepacks QKV into wqkvPacked. 
            // DirectForwardPass expects separate q, k, v weights or a fused one.
            // We'll need to adapt the ForwardPass to handle GGUF's packed format or unpack here.
            // For now, we'll wrap the packed one and hope the engine can be updated.
            weights.put(arch.layerQueryWeight(i), AccelTensor.wrapSegment(lw.wqkvPacked, new long[]{3, (long)lw.wqkvPacked.byteSize() / (3 * 4)}));
            
            weights.put(arch.layerOutputWeight(i), AccelTensor.wrapSegment(lw.wo.segment(), new long[]{lw.wo.numElements()}));
            weights.put(arch.layerFfnNormWeight(i), AccelTensor.wrapSegment(lw.ffnNormWeight, new long[]{(long)lw.ffnNormWeight.byteSize() / 4}));
            
            if (lw.wG != null) weights.put(arch.layerFfnGateWeight(i), AccelTensor.wrapSegment(lw.wG.segment(), new long[]{lw.wG.numElements()}));
            if (lw.wU != null) weights.put(arch.layerFfnUpWeight(i), AccelTensor.wrapSegment(lw.wU.segment(), new long[]{lw.wU.numElements()}));
            if (lw.wD != null) weights.put(arch.layerFfnDownWeight(i), AccelTensor.wrapSegment(lw.wD.segment(), new long[]{lw.wD.numElements()}));
        }

        return weights;
    }

    private static AccelTensor wrap(GGUFModel model, String name) {
        GGUFTensorInfo info = model.tensors().stream()
                .filter(t -> t.name().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Tensor not found in GGUF: " + name));
        
        long[] shape = info.shape();
        // GGUF shapes are usually [dim0, dim1, ...]. 
        // SafeTensor/AccelTensor might expect different ordering.
        return AccelTensor.wrapSegment(model.segment().asSlice(model.dataStart() + info.offset(), info.sizeInBytes()), shape);
    }
}
