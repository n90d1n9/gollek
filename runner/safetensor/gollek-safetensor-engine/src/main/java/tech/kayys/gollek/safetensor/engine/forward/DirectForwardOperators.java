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

record DirectForwardOperators(
        DirectForwardRuntimeContext runtime,
        TraitsResolver traitsResolver) {
    @FunctionalInterface
    interface TraitsResolver {
        ModelConfigTraits resolve(ModelConfig config);
    }

    AccelTensor linear(AccelTensor input,
                       AccelTensor weight,
                       AccelTensor bias,
                       String profileKey,
                       ModelConfig config) {
        return linear(input, weight, bias, profileKey, config, null);
    }

    AccelTensor linear(AccelTensor input,
                       AccelTensor weight,
                       AccelTensor bias,
                       String profileKey,
                       ModelConfig config,
                       AccelTensor outputBuffer) {
        return DirectForwardLinearProjection.linear(new DirectForwardLinearRequest(
                linearContext(config),
                false,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer));
    }

    AccelTensor ffnDownLinear(AccelTensor input,
                              AccelTensor weight,
                              AccelTensor bias,
                              ModelConfig config,
                              String profileKey) {
        return ffnDownLinear(input, weight, bias, config, profileKey, null);
    }

    AccelTensor ffnDownLinear(AccelTensor input,
                              AccelTensor weight,
                              AccelTensor bias,
                              ModelConfig config,
                              String profileKey,
                              AccelTensor outputBuffer) {
        return DirectForwardLinearProjection.ffnDownLinear(new DirectForwardLinearRequest(
                linearContext(config),
                false,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer));
    }

    AccelTensor swigluFfn(AccelTensor input,
                          ModelArchitecture arch,
                          ModelConfig config,
                          AccelTensor gateW,
                          AccelTensor gateB,
                          AccelTensor upW,
                          AccelTensor upB,
                          AccelTensor downW,
                          AccelTensor downB,
                          ForwardWorkspace ws,
                          AccelTensor downOutputBuffer) {
        return DirectForwardGatedFfn.forward(
                runtime,
                traitsResolver.resolve(config),
                config,
                false,
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
    }

    private DirectForwardLinearContext linearContext(ModelConfig config) {
        return new DirectForwardLinearContext(runtime, traitsResolver.resolve(config), config);
    }
}
