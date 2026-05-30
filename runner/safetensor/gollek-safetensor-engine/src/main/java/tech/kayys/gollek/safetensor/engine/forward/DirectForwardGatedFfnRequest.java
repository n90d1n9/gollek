/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

record DirectForwardGatedFfnRequest(
        DirectForwardRuntimeContext runtime,
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
        KVCacheManager.KVCacheSession.ForwardWorkspace ws,
        AccelTensor downOutputBuffer) {

    boolean metalLinearEnabled() {
        return runtime.metalLinearEnabled();
    }

    FFNActivationType activationType() {
        return arch.activationType();
    }
}
