/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardDiagnostics.logSegmentStats;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

final class DirectForwardLayerTailStage {
    private DirectForwardLayerTailStage() {
    }

    static void run(DirectForwardLayerStageContext ctx, AccelTensor perLayerInput) {
        DirectForwardPerLayerInputs.applyResidual(
                ctx.hiddenOut(),
                ctx.hiddenShape(),
                ctx.seqLen(),
                ctx.config(),
                ctx.arch(),
                ctx.layerWeights(),
                perLayerInput,
                ctx.workspace(),
                ctx.useMetalElementwise(),
                ctx.useNativeElementwiseAdd(),
                ctx.addOneRmsNorm(),
                ctx.runtime(),
                ctx.operators());
        applyLayerScalar(ctx);
        if (ctx.verboseLayers() && perLayerInput != null) {
            logSegmentStats(ctx.hiddenOut(), ctx.hiddenShape(), "layer " + ctx.layerIdx() + " postPle");
        }
        if (ctx.verboseLayers() && ctx.layerWeights().hasLayerScalar()) {
            System.err.printf("[DEBUG] layer %d layerScalar=%f%n",
                    ctx.layerIdx(), ctx.layerWeights().layerScalarValue());
        }
    }

    private static void applyLayerScalar(DirectForwardLayerStageContext ctx) {
        if (!ctx.layerWeights().hasLayerScalar()
                || !DirectForwardElementwisePolicy.shouldApplyLayerScalar(ctx.traits())) {
            return;
        }
        DirectForwardElementwiseOps.scaleSegmentInPlace(
                ctx.runtime().log(),
                ctx.runtime().metalBinding(),
                ctx.hiddenOut(),
                ctx.seqLen(),
                ctx.config().hiddenSize(),
                ctx.layerWeights().layerScalarValue(),
                ctx.runtime().canUseMetalLayerScalarScale(ctx.useMetalElementwise(), ctx.seqLen()));
    }
}
