/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

/**
 * Builds the query, key, and value tensors consumed by attention dispatch,
 * preserving reusable workspaces whenever projection and normalization allow it.
 */
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
                !packedQkv && !sharedKv && !useDenseSharedKvState,
                !alternativeAttention);
        AccelTensor packedQkvBuffer = projector.packedQkvProjectionBuffer(in,
                packedQkv && !sharedKv && !useDenseSharedKvState && !alternativeAttention,
                headLayout);

        FlashAttentionProjector.LinearTriple qkvTriple = packedQkv && !sharedKv && !useDenseSharedKvState
                && !alternativeAttention
                ? projector.projectPackedQkv(in, config, modelPolicy, headLayout, packedQkvBuffer)
                : null;
        AccelTensor sharedProjectionOwner = qkvTriple == null ? null : qkvTriple.sharedOwner();
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
        AccelTensor q = null;
        AccelTensor k = null;
        AccelTensor v = null;
        try {
            FlashAttentionProjector.LinearPair qkPair = qkvTriple == null && !packedQkv
                    && (!sharedKv && !useDenseSharedKvState)
                    ? projector.tryMetalHalfLinearPairMixed(
                            in.x, in.qW, in.qB, in.kW, in.kB, "attn_qk_proj_pair", config,
                            modelPolicy,
                            qkvBuffers == null ? null : qkvBuffers.q(),
                            qkvBuffers == null ? null : qkvBuffers.k())
                    : null;

            q = qkvTriple != null ? qkvTriple.first()
                    : (qkPair != null ? qkPair.first() : projector.project(
                            in.x, in.qW, in.qB, "attn_q_proj", config,
                            modelPolicy,
                            qkvBuffers == null ? null : qkvBuffers.q()));
            k = useDenseSharedKvState
                    ? sharedKvState.key()
                    : (sharedKv ? null : (qkvTriple != null ? qkvTriple.second()
                            : (qkPair != null ? qkPair.second() : projector.project(
                                    in.x, in.kW, in.kB, "attn_k_proj", config,
                                    modelPolicy,
                                    qkvBuffers == null ? null : qkvBuffers.k()))));
            v = projectValue(
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
            boolean keySeparatedFromValue = false;
            if (normalizationPolicy.qkNormEnabled() && in.qNormW != null) {
                AccelTensor qNormed = normalizer.perHeadRmsNormReusingInput(q, in.qNormW, config.rmsNormEps(),
                        addOneRmsNorm, modelPolicy);
                if (qNormed != q) {
                    q.close();
                }
                q = qNormed;
            }
            if (normalizationPolicy.qkNormEnabled() && !sharedKv && in.kNormW != null) {
                AccelTensor kNormed = normalizeKey(k, in.kNormW, config.rmsNormEps(), addOneRmsNorm, modelPolicy,
                        alternativeAttention);
                if (kNormed != k) {
                    k.close();
                    keySeparatedFromValue = true;
                }
                k = kNormed;
            }
            if (!sharedKv && normalizationPolicy.valueNormEnabled()) {
                AccelTensor vNormed = normalizeValue(v, config.rmsNormEps(), modelPolicy,
                        alternativeAttention && !keySeparatedFromValue);
                if (vNormed != v) {
                    v.close();
                }
                v = vNormed;
            }
            return new PreparedTensors(q, k, v, sharedProjectionOwner);
        } catch (RuntimeException | Error e) {
            releasePreparedOnFailure(sharedKvState, useDenseSharedKvState, q, k, v, sharedProjectionOwner);
            throw e;
        }
    }

    private void releasePreparedOnFailure(SharedKvState sharedKvState, boolean useDenseSharedKvState,
            AccelTensor q, AccelTensor k, AccelTensor v, AccelTensor sharedProjectionOwner) {
        closeIfOpen(q);
        if (useDenseSharedKvState && sharedKvState != null) {
            sharedKvState.releaseView(k);
            sharedKvState.releaseView(v);
        } else {
            closeIfOpen(k);
            closeIfOpen(v);
        }
        closeIfOpen(sharedProjectionOwner);
    }

    private void closeIfOpen(AccelTensor tensor) {
        if (tensor != null && !tensor.isClosed()) {
            tensor.close();
        }
    }

    private AccelTensor normalizeKey(AccelTensor k, AccelTensor weight, double eps, boolean addOne,
            FlashAttentionModelPolicy modelPolicy, boolean alternativeAttention) {
        if (alternativeAttention) {
            return normalizer.perHeadRmsNorm(k, weight, eps, addOne, modelPolicy);
        }
        return normalizer.perHeadRmsNormReusingInput(k, weight, eps, addOne, modelPolicy);
    }

    private AccelTensor normalizeValue(AccelTensor v, double eps, FlashAttentionModelPolicy modelPolicy,
            boolean aliasesKey) {
        if (aliasesKey) {
            return normalizer.perHeadRmsNormNoWeight(v, eps, modelPolicy);
        }
        return normalizer.perHeadRmsNormNoWeightReusingInput(v, eps, modelPolicy);
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
            return viewKeyAsValueProjection(keyProjection, in.layerIdx);
        }
        if (qkvTriple != null) {
            return qkvTriple.third();
        }
        return projector.project(
                in.x, in.vW, in.vB, "attn_v_proj", config, modelPolicy,
                qkvBuffers == null ? null : qkvBuffers.v());
    }

    private AccelTensor viewKeyAsValueProjection(AccelTensor keyProjection, int layerIdx) {
        if (keyProjection == null) {
            throw new IllegalArgumentException(
                    "Missing attention key projection at layer " + layerIdx
                            + " while deriving Gemma 4 alternative value projection.");
        }
        return AccelTensor.view(keyProjection.dataPtr(), keyProjection.shape(), keyProjection);
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

    record PreparedTensors(AccelTensor query, AccelTensor key, AccelTensor value,
            AccelTensor sharedOwner) implements AutoCloseable {
        @Override
        public void close() {
            closeIfOpen(query);
            closeIfOpen(key);
            closeIfOpen(value);
            closeIfOpen(sharedOwner);
        }

        private static void closeIfOpen(AccelTensor tensor) {
            if (tensor != null && !tensor.isClosed()) {
                tensor.close();
            }
        }
    }
}
