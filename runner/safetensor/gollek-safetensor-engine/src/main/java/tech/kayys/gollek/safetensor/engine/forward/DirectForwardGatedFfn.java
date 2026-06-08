/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardGatedFfn {
    private DirectForwardGatedFfn() {
    }

    static AccelTensor forward(DirectForwardRuntimeContext runtime,
                               ModelConfigTraits traits,
                               ModelConfig config,
                               boolean decodeLogitsPhase,
                               AccelTensor input,
                               ModelArchitecture arch,
                               AccelTensor gateW,
                               AccelTensor gateB,
                               AccelTensor upW,
                               AccelTensor upB,
                               AccelTensor downW,
                               AccelTensor downB,
                               ForwardWorkspace ws,
                               AccelTensor downOutputBuffer) {
        DirectForwardGatedFfnRequest request = new DirectForwardGatedFfnRequest(
                runtime,
                traits,
                config,
                decodeLogitsPhase,
                input,
                arch,
                gateW,
                gateB,
                upW,
                upB,
                downW,
                downB,
                ws,
                downOutputBuffer);

        AccelTensor completeFastPath = DirectForwardGatedFfnFastPaths.tryComplete(request);
        if (completeFastPath != null) {
            return completeFastPath;
        }

        return DirectForwardGatedFfnFallback.forward(request);
    }

}
