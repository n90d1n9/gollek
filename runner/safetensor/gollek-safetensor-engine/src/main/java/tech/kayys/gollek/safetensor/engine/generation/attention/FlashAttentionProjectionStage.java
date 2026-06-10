/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

final class FlashAttentionProjectionStage {
    private final FlashAttentionProjector projector;
    private final FlashAttentionNormalizer normalizer;

    FlashAttentionProjectionStage(FlashAttentionProjector projector, FlashAttentionNormalizer normalizer) {
        this.projector = projector;
        this.normalizer = normalizer;
    }

    PreparedTensors prepare(AttentionInput in, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            FlashAttentionHeadLayout headLayout, FlashAttentionNormalizationPolicy normalizationPolicy,
            SharedKvState sharedKvState, boolean sharedKv, boolean useDenseSharedKvState,
            boolean alternativeAttention) {
        boolean packedQkv = headLayout.packedQkvProjection();
        FlashAttentionProjector.ProjectionBuffers qkvBuffers = projector.attentionProjectionBuffers(in,
                !packedQkv && !sharedKv && !useDenseSharedKvState && !alternativeAttention);

        FlashAttentionProjector.LinearTriple qkvTriple = packedQkv && !sharedKv && !useDenseSharedKvState
                && !alternativeAttention
                ? projector.projectPackedQkv(in, config, modelPolicy, headLayout)
                : null;
        if (qkvTriple == null
                && !packedQkv
                && !sharedKv
                && !useDenseSharedKvState
                && !alternativeAttention) {
            qkvTriple = projector.tryMetalHalfLinearTripleMixed(
                    in.x, in.qW, in.qB, in.kW, in.kB, in.vW, in.vB, "attn_qkv_proj_triple", config,
                    modelPolicy,
                    qkvBuffers == null ? null : qkvBuffers.q(),
                    qkvBuffers == null ? null : qkvBuffers.k(),
                    qkvBuffers == null ? null : qkvBuffers.v());
        }
        FlashAttentionProjector.LinearPair qkPair = qkvTriple == null && !packedQkv
                && (!sharedKv && !useDenseSharedKvState)
                ? projector.tryMetalHalfLinearPairMixed(
                        in.x, in.qW, in.qB, in.kW, in.kB, "attn_qk_proj_pair", config,
                        modelPolicy,
                        qkvBuffers == null ? null : qkvBuffers.q(),
                        qkvBuffers == null ? null : qkvBuffers.k())
                : null;

        AccelTensor q = qkvTriple != null ? qkvTriple.first()
                : (qkPair != null ? qkPair.first() : projector.project(
                        in.x, in.qW, in.qB, "attn_q_proj", config,
                        modelPolicy,
                        qkvBuffers == null ? null : qkvBuffers.q()));
        AccelTensor k = useDenseSharedKvState
                ? sharedKvState.key()
                : (sharedKv ? null : (qkvTriple != null ? qkvTriple.second()
                        : (qkPair != null ? qkPair.second() : projector.project(
                                in.x, in.kW, in.kB, "attn_k_proj", config,
                                modelPolicy,
                                qkvBuffers == null ? null : qkvBuffers.k()))));
        AccelTensor v = projectValue(
                in,
                config,
                modelPolicy,
                sharedKvState,
                sharedKv,
                useDenseSharedKvState,
                alternativeAttention,
                qkvTriple,
                qkvBuffers,
                k);

        q = reshapeQuery(q, in, headLayout, config);
        if (!sharedKv) {
            k = reshapeKey(k, in, headLayout, config);
            v = reshapeValue(v, in, headLayout, config);
        }

        boolean addOneRmsNorm = normalizationPolicy.addOneToRmsNormWeight();
        if (normalizationPolicy.qkNormEnabled() && in.qNormW != null) {
            AccelTensor qNormed = normalizer.perHeadRmsNorm(q, in.qNormW, config.rmsNormEps(), addOneRmsNorm,
                    modelPolicy);
            q.close();
            q = qNormed;
        }
        if (normalizationPolicy.qkNormEnabled() && !sharedKv && in.kNormW != null) {
            AccelTensor kNormed = normalizer.perHeadRmsNorm(k, in.kNormW, config.rmsNormEps(), addOneRmsNorm,
                    modelPolicy);
            k.close();
            k = kNormed;
        }
        if (!sharedKv && normalizationPolicy.valueNormEnabled()) {
            AccelTensor vNormed = normalizer.perHeadRmsNormNoWeight(v, config.rmsNormEps(), modelPolicy);
            v.close();
            v = vNormed;
        }
        return new PreparedTensors(q, k, v);
    }

    private AccelTensor projectValue(AttentionInput in, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            SharedKvState sharedKvState, boolean sharedKv, boolean useDenseSharedKvState,
            boolean alternativeAttention, FlashAttentionProjector.LinearTriple qkvTriple,
            FlashAttentionProjector.ProjectionBuffers qkvBuffers, AccelTensor keyProjection) {
        if (useDenseSharedKvState) {
            return sharedKvState.value();
        }
        if (sharedKv) {
            return null;
        }
        if (alternativeAttention) {
            return copyKeyAsValueProjection(keyProjection, in.layerIdx);
        }
        if (qkvTriple != null) {
            return qkvTriple.third();
        }
        return projector.project(
                in.x, in.vW, in.vB, "attn_v_proj", config, modelPolicy,
                qkvBuffers == null ? null : qkvBuffers.v());
    }

    private AccelTensor copyKeyAsValueProjection(AccelTensor keyProjection, int layerIdx) {
        if (keyProjection == null) {
            throw new IllegalArgumentException(
                    "Missing attention key projection at layer " + layerIdx
                            + " while deriving Gemma 4 alternative value projection.");
        }
        return AccelTensor.copyOf(keyProjection.dataPtr(), keyProjection.shape());
    }

    private AccelTensor reshapeQuery(AccelTensor q, AttentionInput in, FlashAttentionHeadLayout headLayout,
            ModelConfig config) {
        AccelTensor reshaped = FlashAttentionShapeValidator.reshapeProjection(
                q, "query", in.x.size(0), in.x.size(1), headLayout.numQueryHeads(), headLayout.headDim(), config,
                in.layerIdx);
        q.close();
        return reshaped;
    }

    private AccelTensor reshapeKey(AccelTensor k, AttentionInput in, FlashAttentionHeadLayout headLayout,
            ModelConfig config) {
        AccelTensor reshaped = FlashAttentionShapeValidator.reshapeProjection(
                k, "key", in.x.size(0), in.x.size(1), headLayout.numKeyValueHeads(), headLayout.headDim(), config,
                in.layerIdx);
        k.close();
        return reshaped;
    }

    private AccelTensor reshapeValue(AccelTensor v, AttentionInput in, FlashAttentionHeadLayout headLayout,
            ModelConfig config) {
        AccelTensor reshaped = FlashAttentionShapeValidator.reshapeProjection(
                v, "value", in.x.size(0), in.x.size(1), headLayout.numKeyValueHeads(), headLayout.headDim(), config,
                in.layerIdx);
        v.close();
        return reshaped;
    }

    record PreparedTensors(AccelTensor query, AccelTensor key, AccelTensor value) {
    }
}
