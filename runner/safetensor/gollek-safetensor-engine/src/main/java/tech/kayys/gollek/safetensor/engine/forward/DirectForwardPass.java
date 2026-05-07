/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.engine.generation.attention.FlashAttentionKernel;
import tech.kayys.gollek.safetensor.engine.generation.moe.MoeForwardPass;
import tech.kayys.gollek.metal.MetalComputeBackend;
import jakarta.enterprise.inject.Instance;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;

/**
 * Full transformer forward pass using AccelTensor + Apple Accelerate.
 * No LibTorch dependency.
 */
@ApplicationScoped
public class DirectForwardPass {
    private static final Logger log = Logger.getLogger(DirectForwardPass.class);
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";
    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY = "gollek.safetensor.experimental_metal_linear";
    private static final float GELU_INNER_SCALE = 0.79788456f;
    private static final float GELU_CUBIC_COEFF = 0.044715f;
    @Inject
    FlashAttentionKernel attentionKernel;
    @Inject
    MoeForwardPass moeForwardPass;
    @Inject
    Instance<MetalComputeBackend> metalBackendInstance;
    
    private MetalComputeBackend metal;

    @jakarta.annotation.PostConstruct
    void init() {
        if (metalBackendInstance.isResolvable()) {
            this.metal = metalBackendInstance.get();
        }
    }

    public float[] prefill(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(inputIds, weights, config, arch, kvCache);
        float[] result = materializeLogits(logits);
        logPrefillDiagnostics(result, config);
        return result;
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        return prefill(embeddings, inputIds, null, weights, config, arch, kvCache);
    }

    public float[] prefill(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = prefillLogitsTensor(embeddings, inputIds, perLayerInputs, weights, config, arch, kvCache);
        return materializeLogits(logits);
    }

    public AccelTensor prefillLogitsTensor(long[] inputIds, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight: " + arch.embedTokensWeight());
        AccelTensor embedded = embeddingLookup(embedTable, inputIds);
        float scale = arch.embeddingScaleFactor((int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        AccelTensor[] perLayerInputs = buildPerLayerInputs(inputIds, hidden, weights, config, arch);
        try {
            if (scale != 1.0f)
                embedded.close();
            return prefillLogitsTensor(hidden, inputIds, perLayerInputs, weights, config, arch, kvCache);
        } finally {
            closePerLayerInputs(perLayerInputs);
            hidden.closeWithParent();
        }
    }

    public AccelTensor prefillLogitsTensor(AccelTensor embeddings, long[] inputIds, AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache) {
        long seqLen = embeddings.size(1);
        if (seqLen < 1) {
            throw new IllegalArgumentException("Invalid sequence length: " + seqLen + ". Prompt must result in at least one token.");
        }
        KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
        ws.ensureCapacity((long) seqLen * config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

        // Initial copy: embeddings -> hiddenASeg
        java.lang.foreign.MemorySegment.copy(embeddings.dataPtr(), 0, ws.getHiddenASeg(), 0, (long) seqLen * config.hiddenSize() * 4);

        java.lang.foreign.MemorySegment currentHidden = ws.getHiddenASeg();
        java.lang.foreign.MemorySegment nextHidden = ws.getHiddenBSeg();

        boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            if (verbose) {
                System.err.printf("[DEBUG] Prefill Layer %d/%d start\n", i, config.numHiddenLayers());
                System.err.flush();
            }
            transformerLayer(currentHidden, nextHidden, inputIds, perLayerInputs != null ? perLayerInputs[i] : null,
                    weights, config, arch, kvCache, i, 0, (int) seqLen, ws);

            // Swap buffers
            java.lang.foreign.MemorySegment temp = currentHidden;
            currentHidden = nextHidden;
            nextHidden = temp;
        }

        AccelTensor hidden = AccelTensor.view(currentHidden, embeddings.shape());
        
        AccelTensor normed;
        if (canUseMetal()) {
            normed = AccelTensor.view(ws.getNormedAttnSeg(), hidden.shape());
            rmsNormRowsMetal(normed.dataPtr(), hidden.dataPtr(),
                    weights.get(arch.finalNormWeight()).dataPtr(), (int) seqLen, config.hiddenSize(),
                    (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        } else {
            normed = AccelOps.rmsNorm(hidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                    useAddOneRmsNorm(arch, config));
        }
        if (hidden != embeddings)
            hidden.close();

        AccelTensor lastPos = selectLastToken(normed, (int) seqLen);
        if (normed.dataPtr() != ws.getNormedAttnSeg())
            normed.close();

        AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
        if (lmHeadW == null && config.tieWordEmbeddings()) {
            lmHeadW = weights.get(arch.embedTokensWeight()); // weight tying
        }
        if (lmHeadW == null) {
            throw new IllegalStateException(
                    "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
        }
        long tLogits0 = System.nanoTime();
        AccelTensor logits = linear(lastPos, lmHeadW, null);
        DirectInferenceEngine.recordLogitsProjectionNanos(System.nanoTime() - tLogits0);
        lastPos.close();
        kvCache.advance((int) seqLen);
        return logits;
    }

    public float[] decode(long tokenId, int startPos, Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor logits = decodeLogitsTensor(tokenId, startPos, weights, config, arch, kvCache);
        try {
            long tLogitsCopy0 = System.nanoTime();
            float[] result = logits.toFloatArray();
            DirectInferenceEngine.recordLogitsMaterializationNanos(System.nanoTime() - tLogitsCopy0);
            return result;
        } finally {
            logits.close();
        }
    }

    public AccelTensor decodeLogitsTensor(long tokenId, int startPos, Map<String, AccelTensor> weights,
            ModelConfig config, ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache) {
        AccelTensor embedTable = weights.get(arch.embedTokensWeight());
        if (embedTable == null)
            throw new IllegalStateException("Missing embed tokens weight.");
        AccelTensor embedded = embeddingLookup(embedTable, new long[] { tokenId });
        float scale = arch.embeddingScaleFactor((int) embedded.size(-1));
        AccelTensor hidden = scale != 1.0f ? AccelOps.mulScalar(embedded, scale) : embedded;
        AccelTensor[] perLayerInputs = buildPerLayerInputs(new long[] { tokenId }, hidden, weights, config, arch);
        try {
            if (scale != 1.0f)
                embedded.close();
            long[] tokenIds = new long[] { tokenId };

            KVCacheManager.KVCacheSession.ForwardWorkspace ws = kvCache.getWorkspace();
            ws.ensureCapacity(config.hiddenSize(), config.hiddenSize(), config.intermediateSize());

            // Initial copy: hidden -> hiddenASeg
            java.lang.foreign.MemorySegment.copy(hidden.dataPtr(), 0, ws.getHiddenASeg(), 0, config.hiddenSize() * 4);
            hidden.close();

            java.lang.foreign.MemorySegment currentHidden = ws.getHiddenASeg();
            java.lang.foreign.MemorySegment nextHidden = ws.getHiddenBSeg();

            for (int i = 0; i < config.numHiddenLayers(); i++) {
                transformerLayer(currentHidden, nextHidden, tokenIds, perLayerInputs != null ? perLayerInputs[i] : null,
                        weights, config, arch, kvCache, i, startPos, 1, ws);
                // Swap
                java.lang.foreign.MemorySegment temp = currentHidden;
                currentHidden = nextHidden;
                nextHidden = temp;
            }

            AccelTensor finalHidden = AccelTensor.view(currentHidden, new long[]{1, 1, config.hiddenSize()});
            
            AccelTensor normed;
            if (canUseMetal()) {
                normed = AccelTensor.view(ws.getNormedAttnSeg(), finalHidden.shape());
                rmsNormRowsMetal(normed.dataPtr(), finalHidden.dataPtr(),
                        weights.get(arch.finalNormWeight()).dataPtr(), 1, config.hiddenSize(),
                        (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            } else {
                normed = AccelOps.rmsNorm(finalHidden, weights.get(arch.finalNormWeight()), config.rmsNormEps(),
                        useAddOneRmsNorm(arch, config));
            }
            finalHidden.close();

            AccelTensor lmHeadW = weights.get(arch.lmHeadWeight());
            if (lmHeadW == null && config.tieWordEmbeddings()) {
                lmHeadW = weights.get(arch.embedTokensWeight());
            }
            if (lmHeadW == null) {
                throw new IllegalStateException(
                        "Missing lm_head weight. Safetensor file might be incomplete or config.tie_word_embeddings is missing.");
            }
            long tLogits0 = System.nanoTime();
            AccelTensor logits = linear(normed, lmHeadW, null);
            DirectInferenceEngine.recordLogitsProjectionNanos(System.nanoTime() - tLogits0);
            if (normed.dataPtr() != ws.getNormedAttnSeg())
                normed.close();
            kvCache.advance(1);
            return logits;
        } finally {
            closePerLayerInputs(perLayerInputs);
            if (!hidden.isClosed())
                hidden.closeWithParent();
        }
    }

    private float[] materializeLogits(AccelTensor logits) {
        try {
            long tLogitsCopy0 = System.nanoTime();
            float[] result = logits.toFloatArray();
            DirectInferenceEngine.recordLogitsMaterializationNanos(System.nanoTime() - tLogitsCopy0);
            return result;
        } finally {
            logits.close();
        }
    }

    private void logPrefillDiagnostics(float[] result, ModelConfig config) {
        boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
        if (!verbose || config.numAttentionHeads() <= 0) {
            return;
        }

        double sum = 0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float f : result) {
            sum += f;
            if (f < min) {
                min = f;
            }
            if (f > max) {
                max = f;
            }
        }

        System.err.printf("[DEBUG] Logits stats: min=%f, max=%f, sum=%f, size=%d%n", min, max, sum, result.length);

        List<Integer> topIndices = new ArrayList<>();
        for (int i = 0; i < result.length; i++) {
            topIndices.add(i);
        }
        topIndices.sort((a, b) -> Float.compare(result[b], result[a]));
        for (int k = 0; k < Math.min(5, topIndices.size()); k++) {
            int id = topIndices.get(k);
            System.err.printf("  Top %d: ID=%d, val=%f%n", k, id, result[id]);
        }
    }

    private void transformerLayer(java.lang.foreign.MemorySegment hiddenIn, java.lang.foreign.MemorySegment hiddenOut, 
            long[] inputIds, AccelTensor perLayerInput, Map<String, AccelTensor> weights, ModelConfig config, 
            ModelArchitecture arch, KVCacheManager.KVCacheSession kvCache, int layerIdx, int startPos, int seqLen, 
            KVCacheManager.KVCacheSession.ForwardWorkspace ws) {
        
        long[] hiddenShape = new long[]{1, seqLen, config.hiddenSize()};
        boolean verbose = "true".equals(System.getProperty("gollek.verbose"));
        
        // Attention norm
        java.lang.foreign.MemorySegment normedAttnSeg = ws.getNormedAttnSeg();
        if (canUseMetal()) {
            rmsNormRowsMetal(normedAttnSeg, hiddenIn,
                    weights.get(arch.layerAttentionNormWeight(layerIdx)).dataPtr(), seqLen, config.hiddenSize(),
                    (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        } else {
            // Fallback (slow but stable)
            AccelTensor in = AccelTensor.view(hiddenIn, hiddenShape);
            AccelTensor out = AccelOps.rmsNorm(in, weights.get(arch.layerAttentionNormWeight(layerIdx)), config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            java.lang.foreign.MemorySegment.copy(out.dataPtr(), 0, normedAttnSeg, 0, (long) seqLen * config.hiddenSize() * 4);
            out.close();
        }

        FlashAttentionKernel.AttentionInput attnIn = new FlashAttentionKernel.AttentionInput(
                AccelTensor.view(normedAttnSeg, hiddenShape),
                weights.get(arch.layerQueryWeight(layerIdx)),
                weights.get(arch.layerKeyWeight(layerIdx)),
                weights.get(arch.layerValueWeight(layerIdx)),
                weights.get(arch.layerOutputWeight(layerIdx)),
                resolveBias(weights, arch.layerQueryBias(layerIdx)),
                resolveBias(weights, arch.layerKeyBias(layerIdx)),
                resolveBias(weights, arch.layerValueBias(layerIdx)),
                weights.get(arch.layerOutputBias(layerIdx)),
                arch, config, kvCache, layerIdx, startPos,
                /* isCausal= */ true,
                weights.get(arch.layerQueryNormWeight(layerIdx)),
                weights.get(arch.layerKeyNormWeight(layerIdx)),
                weights.get(arch.layerPostAttnNormWeight(layerIdx)));

        if (verbose) {
            System.err.printf("[DEBUG] Layer %d Attention start\n", layerIdx);
            System.err.flush();
        }
        long tAttention0 = System.nanoTime();
        AccelTensor attnOut = attentionKernel.compute(attnIn);
        DirectInferenceEngine.recordAttentionNanos(System.nanoTime() - tAttention0);
        if (verbose) {
            logTensorStats(attnOut, "layer " + layerIdx + " attnOut");
        }

        // First residual add after attention.
        residualAdd(hiddenIn, attnOut, hiddenOut, hiddenShape, seqLen, config.hiddenSize());
        attnOut.close();
        if (verbose) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postAttnResidual");
        }

        // MLP input norm.
        AccelTensor preFfnNormW = weights.get(arch.layerPreFfnNormWeight(layerIdx));
        AccelTensor postFfnNormW = weights.get(arch.layerFfnNormWeight(layerIdx));
        if (preFfnNormW == null) preFfnNormW = postFfnNormW;

        java.lang.foreign.MemorySegment normedFfnSeg = ws.getNormedFfnSeg();
        if (canUseMetal()) {
            rmsNormRowsMetal(normedFfnSeg, hiddenOut, preFfnNormW.dataPtr(), seqLen, config.hiddenSize(),
                    (float) config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        } else {
            AccelTensor in = AccelTensor.view(hiddenOut, hiddenShape);
            AccelTensor out = AccelOps.rmsNorm(in, preFfnNormW, config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            java.lang.foreign.MemorySegment.copy(out.dataPtr(), 0, normedFfnSeg, 0, (long) seqLen * config.hiddenSize() * 4);
            out.close();
        }

        long tFfn0 = System.nanoTime();
        AccelTensor mlpOut = swigluFfn(AccelTensor.view(normedFfnSeg, hiddenShape), arch,
                weights.get(arch.layerFfnGateWeight(layerIdx)), weights.get(arch.layerFfnGateBias(layerIdx)),
                weights.get(arch.layerFfnUpWeight(layerIdx)), weights.get(arch.layerFfnUpBias(layerIdx)),
                weights.get(arch.layerFfnDownWeight(layerIdx)), weights.get(arch.layerFfnDownBias(layerIdx)), ws);
        DirectInferenceEngine.recordFfnNanos(System.nanoTime() - tFfn0);
        if (verbose) {
            logTensorStats(mlpOut, "layer " + layerIdx + " mlpOut");
        }

        // Post-FFN norm before the second residual add.
        AccelTensor mlpNormed;
        if (postFfnNormW != null) {
            mlpNormed = AccelOps.rmsNorm(mlpOut, postFfnNormW, config.rmsNormEps(), useAddOneRmsNorm(arch, config));
            mlpOut.close();
        } else {
            mlpNormed = mlpOut;
        }

        AccelTensor layerScalar = weights.get(arch.layerScalarWeight(layerIdx));
        residualAdd(hiddenOut, mlpNormed, hiddenOut, hiddenShape, seqLen, config.hiddenSize());
        mlpNormed.close();
        if (verbose) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postFfnResidual");
        }

        applyPerLayerResidual(hiddenOut, hiddenShape, seqLen, config, arch, weights, layerIdx, perLayerInput);
        if (layerScalar != null) {
            float outputScale = readScalarValue(layerScalar);
            scaleSegmentInPlace(hiddenOut, hiddenShape, seqLen, config.hiddenSize(), outputScale);
        }
        if (verbose && perLayerInput != null) {
            logSegmentStats(hiddenOut, hiddenShape, "layer " + layerIdx + " postPle");
        }
        if (verbose && layerScalar != null) {
            float skipScale = readScalarValue(layerScalar);
            System.err.printf("[DEBUG] layer %d layerScalar=%f%n", layerIdx, skipScale);
        }
    }

    private void checkStability(AccelTensor t, String label) {
        float[] arr = t.toFloatArray();
        int nan = 0;
        int inf = 0;
        for (float f : arr) {
            if (Float.isNaN(f))
                nan++;
            if (Float.isInfinite(f))
                inf++;
        }
        if (nan > 0 || inf > 0) {
        }
    }

    private void logTensorStats(AccelTensor t, String label) {
        logArrayStats(t.toFloatArray(), label);
    }

    private void logSegmentStats(java.lang.foreign.MemorySegment seg, long[] shape, String label) {
        logTensorStats(AccelTensor.view(seg, shape), label);
    }

    private void logArrayStats(float[] arr, String label) {
        double sum = 0.0;
        double sumSq = 0.0;
        double sumAbs = 0.0;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float v : arr) {
            sum += v;
            sumSq += (double) v * v;
            sumAbs += Math.abs(v);
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double n = Math.max(1, arr.length);
        double mean = sum / n;
        double rms = Math.sqrt(sumSq / n);
        double meanAbs = sumAbs / n;
        System.err.printf("[DEBUG] %s stats: mean=%f meanAbs=%f rms=%f min=%f max=%f size=%d%n",
                label, mean, meanAbs, rms, min, max, arr.length);
        System.err.flush();
    }

    AccelTensor ffnNonGated(AccelTensor x, AccelTensor upW, AccelTensor downW) {
        AccelTensor up = AccelOps.linear(x, upW);
        AccelTensor act = AccelOps.silu(up);
        up.close();
        AccelTensor out = AccelOps.linear(act, downW);
        act.close();
        return out;
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB) {
        return swigluFfn(x, arch, gateW, gateB, upW, upB, downW, downB, null);
    }

    public AccelTensor swigluFfn(AccelTensor x, ModelArchitecture arch, AccelTensor gateW, AccelTensor gateB, AccelTensor upW, AccelTensor upB,
            AccelTensor downW, AccelTensor downB, KVCacheManager.KVCacheSession.ForwardWorkspace ws) {
        AccelTensor gate = linear(x, gateW, gateB);
        AccelTensor up = linear(x, upW, upB);

        AccelTensor combined;
        if (arch.activationType() == FFNActivationType.GELU) {
            if (ws != null) {
                combined = AccelTensor.view(ws.getCombinedSeg(), gate.shape());
                fusedGeglu(combined, gate, up);
            } else {
                AccelTensor activated = AccelOps.gelu(gate);
                gate.close();
                combined = AccelOps.mul(activated, up);
                activated.close();
            }
        } else if (canUseMetal() && ws != null) {
            combined = AccelTensor.view(ws.getCombinedSeg(), gate.shape());
            try {
                metal.siluFfn(combined.dataPtr(), gate.dataPtr(), up.dataPtr(),
                        (int) gate.numel());
            } catch (IllegalStateException | UnsupportedOperationException e) {
                combined = AccelOps.swiglu(gate, up);
            }
        } else {
            combined = AccelOps.swiglu(gate, up);
        }

        if (!gate.isClosed()) gate.close();
        up.close();
        AccelTensor out = linear(combined, downW, downB);
        if (ws == null || combined.dataPtr() != ws.getCombinedSeg())
            combined.close();
        return out;
    }

    /**
     * In-place GeGLU combine for Gemma-style FFNs.
     * Writes gelu(gate) * up directly into the reusable workspace buffer.
     */
    private void fusedGeglu(AccelTensor out, AccelTensor gate, AccelTensor up) {
        java.lang.foreign.MemorySegment outSeg = out.dataPtr();
        java.lang.foreign.MemorySegment gateSeg = gate.dataPtr();
        java.lang.foreign.MemorySegment upSeg = up.dataPtr();
        long n = gate.numel();

        long i = 0;
        long upperBound = FLOAT_SPECIES.loopBound(n);
        FloatVector half = FloatVector.broadcast(FLOAT_SPECIES, 0.5f);
        FloatVector one = FloatVector.broadcast(FLOAT_SPECIES, 1.0f);
        FloatVector innerScale = FloatVector.broadcast(FLOAT_SPECIES, GELU_INNER_SCALE);
        FloatVector cubicCoeff = FloatVector.broadcast(FLOAT_SPECIES, GELU_CUBIC_COEFF);

        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector g = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, gateSeg, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector u = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, upSeg, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector g2 = g.mul(g);
            FloatVector g3 = g2.mul(g);
            FloatVector inner = g.add(g3.mul(cubicCoeff)).mul(innerScale);
            FloatVector gelu = g.mul(half).mul(one.add(tanhApprox(inner)));
            gelu.mul(u).intoMemorySegment(outSeg, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }

        for (; i < n; i++) {
            float g = gateSeg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float u = upSeg.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            float inner = GELU_INNER_SCALE * (g + GELU_CUBIC_COEFF * g * g * g);
            float gelu = 0.5f * g * (1.0f + (float) Math.tanh(inner));
            outSeg.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, gelu * u);
        }
    }

    private static FloatVector tanhApprox(FloatVector x) {
        FloatVector expNeg2x = expApprox(x.mul(-2.0f));
        return FloatVector.broadcast(FLOAT_SPECIES, 2.0f)
                .div(FloatVector.broadcast(FLOAT_SPECIES, 1.0f).add(expNeg2x))
                .sub(FloatVector.broadcast(FLOAT_SPECIES, 1.0f));
    }

    private static FloatVector expApprox(FloatVector x) {
        FloatVector clamped = x.max(-20.0f).min(20.0f);
        FloatVector x2 = clamped.mul(clamped);
        FloatVector x3 = x2.mul(clamped);
        FloatVector x4 = x3.mul(clamped);
        FloatVector x5 = x4.mul(clamped);
        FloatVector x6 = x5.mul(clamped);

        return FloatVector.broadcast(FLOAT_SPECIES, 1.0f)
                .add(clamped)
                .add(x2.mul(1.0f / 2.0f))
                .add(x3.mul(1.0f / 6.0f))
                .add(x4.mul(1.0f / 24.0f))
                .add(x5.mul(1.0f / 120.0f))
                .add(x6.mul(1.0f / 720.0f));
    }

    // ── Embedding ─────────────────────────────────────────────────────

    public AccelTensor embeddingLookup(AccelTensor embedTable, long[] tokenIds) {
        int seqLen = tokenIds.length;
        AccelTensor selected = embedTable.indexSelect(tokenIds);
        long hiddenSize = selected.size(1);
        return selected.reshape(1L, seqLen, hiddenSize);
    }

    private AccelTensor selectLastToken(AccelTensor hidden, int seqLen) {
        if (seqLen < 1) return hidden; // Safety fallback
        long hiddenSize = hidden.size(hidden.shape().length - 1);
        long lastTokenOffset = Math.max(0L, (long) (seqLen - 1) * hiddenSize);
        return AccelTensor.view(hidden.dataPtr().asSlice(lastTokenOffset * 4L), new long[] { 1, hiddenSize });
    }

    // ── Linear ────────────────────────────────────────────────────────

    private AccelTensor linear(AccelTensor input, AccelTensor weight, AccelTensor bias) {
        if (weight.isQuantized()) {
            AccelTensor dequantized = weight.dequantize();
            try {
                return linear(input, dequantized, bias);
            } finally {
                if (dequantized != weight && !dequantized.isClosed()) {
                    dequantized.close();
                }
            }
        }
        AccelTensor mm = AccelOps.linear(input, weight);
        if (bias != null) {
            AccelTensor out = AccelOps.add(mm, bias);
            mm.close();
            return out;
        }
        return mm;
    }

    private boolean canUseExperimentalMetalLinear() {
        return canUseMetal() && Boolean.getBoolean(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
    }

    private static AccelTensor resolveBias(Map<String, AccelTensor> weights, String key) {
        return key != null ? weights.get(key) : null;
    }

    private boolean canUseMetal() {
        if (Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY)) {
            return false;
        }
        return metal != null && metal.deviceName() != null && !metal.deviceName().contains("CPU");
    }

    private void rmsNormRowsMetal(java.lang.foreign.MemorySegment out, java.lang.foreign.MemorySegment in,
            java.lang.foreign.MemorySegment weight, int rows, int hiddenSize, float eps, boolean addOne) {
        long rowBytes = (long) hiddenSize * Float.BYTES;
        for (int row = 0; row < rows; row++) {
            long offset = (long) row * rowBytes;
            metal.rmsNorm(out.asSlice(offset, rowBytes), in.asSlice(offset, rowBytes), weight, hiddenSize, eps, addOne);
        }
    }

    private void residualAdd(java.lang.foreign.MemorySegment left, AccelTensor right,
            java.lang.foreign.MemorySegment out, long[] shape, int seqLen, long hiddenSize) {
        if (canUseMetal()) {
            try {
                metal.add(left, right.dataPtr(), out, shape);
                return;
            } catch (IllegalStateException | UnsupportedOperationException e) {
                log.debugf("Falling back from Metal add to direct CPU accumulation: %s", e.getMessage());
            }
        }

        addSegments(left, right.dataPtr(), out, seqLen * hiddenSize);
    }

    private void addSegments(java.lang.foreign.MemorySegment left, java.lang.foreign.MemorySegment right,
            java.lang.foreign.MemorySegment out, long elements) {
        long i = 0;
        long upperBound = FLOAT_SPECIES.loopBound(elements);
        for (; i < upperBound; i += FLOAT_SPECIES.length()) {
            FloatVector leftVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, left, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            FloatVector rightVec = FloatVector.fromMemorySegment(
                    FLOAT_SPECIES, right, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
            leftVec.add(rightVec).intoMemorySegment(out, i * Float.BYTES, java.nio.ByteOrder.nativeOrder());
        }
        for (; i < elements; i++) {
            float sum = left.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i)
                    + right.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, sum);
        }
    }

    private AccelTensor[] buildPerLayerInputs(long[] inputIds, AccelTensor inputsEmbeds,
            Map<String, AccelTensor> weights, ModelConfig config, ModelArchitecture arch) {
        if (config.hiddenSizePerLayerInput() <= 0) {
            return null;
        }

        AccelTensor packedPleEmbeddings = weights.get(arch.embedTokensPerLayerWeight());
        AccelTensor pleProjection = weights.get(arch.perLayerModelProjectionWeight());
        AccelTensor pleProjectionNorm = weights.get(arch.perLayerProjectionNormWeight());
        if (packedPleEmbeddings == null || pleProjection == null || pleProjectionNorm == null) {
            return null;
        }

        int numLayers = config.numHiddenLayers();
        int pleDim = config.hiddenSizePerLayerInput();
        int seqLen = inputIds.length;

        AccelTensor tokenPleRaw = packedPleEmbeddings.indexSelect(inputIds)
                .reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor tokenPle = AccelOps.mulScalar(tokenPleRaw, (float) Math.sqrt(Math.max(1, pleDim)));
        tokenPleRaw.close();
        AccelTensor projectedPle = AccelOps.linear(inputsEmbeds, pleProjection);
        AccelTensor projectedPleScaled = AccelOps.mulScalar(projectedPle,
                (float) (1.0 / Math.sqrt(Math.max(1, config.hiddenSize()))));
        projectedPle.close();

        AccelTensor projectedPle4d = projectedPleScaled.reshape(1L, seqLen, numLayers, pleDim);
        AccelTensor projectedPleNormed = AccelOps.rmsNorm(projectedPle4d, pleProjectionNorm,
                config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        projectedPle4d.close();

        AccelTensor combinedPle = AccelOps.add(projectedPleNormed, tokenPle);
        projectedPleNormed.close();
        tokenPle.close();

        AccelTensor scaledPle = AccelOps.mulScalar(combinedPle, (float) (1.0 / Math.sqrt(2.0)));
        combinedPle.close();

        AccelTensor[] layers = new AccelTensor[numLayers];
        java.lang.foreign.MemorySegment src = scaledPle.dataPtr();
        for (int layer = 0; layer < numLayers; layer++) {
            AccelTensor layerTensor = AccelTensor.zeros(new long[] { 1L, seqLen, pleDim });
            java.lang.foreign.MemorySegment dst = layerTensor.dataPtr();
            for (int token = 0; token < seqLen; token++) {
                long srcIndex = ((long) token * numLayers + layer) * pleDim;
                long dstIndex = (long) token * pleDim;
                java.lang.foreign.MemorySegment.copy(src, srcIndex * Float.BYTES, dst, dstIndex * Float.BYTES,
                        (long) pleDim * Float.BYTES);
            }
            layers[layer] = layerTensor;
        }
        scaledPle.close();
        return layers;
    }

    private void applyPerLayerResidual(java.lang.foreign.MemorySegment hiddenSeg, long[] hiddenShape, int seqLen,
            ModelConfig config, ModelArchitecture arch, Map<String, AccelTensor> weights, int layerIdx,
            AccelTensor perLayerInput) {
        if (perLayerInput == null) {
            return;
        }

        AccelTensor gateWeight = weights.get(arch.layerPerLayerInputGateWeight(layerIdx));
        AccelTensor projectionWeight = weights.get(arch.layerPerLayerProjectionWeight(layerIdx));
        AccelTensor normWeight = weights.get(arch.layerPostPerLayerInputNormWeight(layerIdx));
        if (gateWeight == null || projectionWeight == null || normWeight == null) {
            return;
        }

        AccelTensor hidden = AccelTensor.view(hiddenSeg, hiddenShape);
        AccelTensor gate = AccelOps.linear(hidden, gateWeight);
        if (arch.activationType() == FFNActivationType.GELU) {
            AccelTensor activated = AccelOps.gelu(gate);
            gate.close();
            gate = activated;
        } else {
            AccelTensor activated = AccelOps.silu(gate);
            gate.close();
            gate = activated;
        }
        AccelTensor mixed = AccelOps.mul(gate, perLayerInput);
        gate.close();

        AccelTensor projected = AccelOps.linear(mixed, projectionWeight);
        mixed.close();

        AccelTensor normed = AccelOps.rmsNorm(projected, normWeight,
                config.rmsNormEps(), useAddOneRmsNorm(arch, config));
        projected.close();

        AccelTensor residual = AccelOps.add(hidden, normed);
        normed.close();
        java.lang.foreign.MemorySegment.copy(residual.dataPtr(), 0, hiddenSeg, 0,
                (long) seqLen * config.hiddenSize() * Float.BYTES);
        residual.close();
    }

    private void closePerLayerInputs(AccelTensor[] perLayerInputs) {
        if (perLayerInputs == null) {
            return;
        }
        for (AccelTensor perLayerInput : perLayerInputs) {
            if (perLayerInput != null) {
                perLayerInput.close();
            }
        }
    }

    private boolean useAddOneRmsNorm(ModelArchitecture arch, ModelConfig config) {
        return arch.addOneToRmsNormWeight() && !isGemma4Text(config);
    }

    private boolean isGemma4Text(ModelConfig config) {
        String modelType = config != null && config.modelType() != null ? config.modelType().toLowerCase() : "";
        return modelType.startsWith("gemma4");
    }

    private void scaleSegmentInPlace(java.lang.foreign.MemorySegment seg, long[] shape, int seqLen, long hiddenSize, float scale) {
        AccelTensor view = AccelTensor.view(seg, shape);
        AccelTensor scaled = AccelOps.mulScalar(view, scale);
        java.lang.foreign.MemorySegment.copy(scaled.dataPtr(), 0, seg, 0, (long) seqLen * hiddenSize * 4);
        scaled.close();
    }

    private float readScalarValue(AccelTensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("Tensor is null");
        }
        if (tensor.quantType() == AccelTensor.QuantType.F32) {
            return tensor.dataPtr().get(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        }
        AccelTensor dequantized = tensor.dequantizeTransient();
        try {
            return dequantized.dataPtr().get(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
        } finally {
            if (dequantized != tensor && !dequantized.isClosed()) {
                dequantized.close();
            }
        }
    }

}
