package tech.kayys.gollek.gguf.model.aljabr;

import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Module;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.WeightAdapter;

public class VisionProjector extends Module {
    private final Linear proj;

    public VisionProjector(WeightAdapter weights) {
        // PaliGemma uses a single Linear projection (often bias=true)
        Tensor wProj = weights.getWeight("mm_proj.weight");
        Tensor bProj = weights.getWeight("mm_proj.bias");
        
        proj = new Linear(wProj, bProj);
        registerModule("proj", proj);
    }
    
    @Override
    public Tensor forward(Tensor x) {
        // x: [batch, num_patches, embed_dim]
        // PaliGemma usually applies the projector directly to the normalized patch embeddings
        return proj.forward(x);
    }
}
