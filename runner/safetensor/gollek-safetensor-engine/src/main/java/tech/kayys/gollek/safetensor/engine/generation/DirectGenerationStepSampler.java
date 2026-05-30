/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.canUseDirectGreedySampling;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.initializePromptFrequencies;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.sampleGreedyFromTensor;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.sampleNextToken;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;

final class DirectGenerationStepSampler {
    private final Supplier<DirectForwardPass> forwardPass;
    private final Supplier<TokenSampler> tokenSampler;

    DirectGenerationStepSampler(Supplier<DirectForwardPass> forwardPass, Supplier<TokenSampler> tokenSampler) {
        this.forwardPass = forwardPass;
        this.tokenSampler = tokenSampler;
    }

    SamplingState createState(DirectLoadedModel model, GenerationConfig cfg, long[] promptTokenIds, Set<Integer> stops,
            SamplingMode mode) {
        boolean directGreedy = canUseDirectGreedySampling(cfg);
        int[] frequencies = directGreedy ? null : initializePromptFrequencies(model.config(), promptTokenIds);
        Random rng = directGreedy ? null : new Random();
        GreedySamplingMasks greedyMasks = directGreedy && mode == SamplingMode.TOKENIZER_AWARE
                ? GenerationTokenPolicy.greedySamplingMasksFor(model.baseGreedySamplingMasks(), model.tokenizer(),
                        stops, model.config().vocabSize(), cfg, model.runtimeTraits())
                : null;
        return new SamplingState(mode, directGreedy, frequencies, rng, stops, greedyMasks);
    }

    StepResult prefill(long[] inputIds, DirectLoadedModel model, KVCacheManager.KVCacheSession session,
            GenerationConfig cfg, SamplingState sampling) {
        ModelConfig config = model.config();
        ModelArchitecture arch = model.architecture();
        long tForward0 = System.nanoTime();
        if (sampling.directGreedy()) {
            AccelTensor logits = forwardPass.get().prefillLogitsTensor(inputIds, model.weights(), config, arch,
                    session);
            long forwardNanos = System.nanoTime() - tForward0;
            long tSample0 = System.nanoTime();
            int next = sampleGreedy(logits, model, sampling, true);
            return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
        }

        float[] logits = forwardPass.get().prefill(inputIds, model.weights(), config, arch, session);
        long forwardNanos = System.nanoTime() - tForward0;
        long tSample0 = System.nanoTime();
        int next = sampleLogits(logits, model, cfg, sampling, true);
        return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
    }

    StepResult decode(int previousToken, int decodeStartPos, DirectLoadedModel model,
            KVCacheManager.KVCacheSession session, GenerationConfig cfg, SamplingState sampling) {
        ModelConfig config = model.config();
        ModelArchitecture arch = model.architecture();
        long tForward0 = System.nanoTime();
        if (sampling.directGreedy()) {
            AccelTensor logits = forwardPass.get().decodeLogitsTensor(previousToken, decodeStartPos, model.weights(),
                    config, arch, session, true);
            long forwardNanos = System.nanoTime() - tForward0;
            long tSample0 = System.nanoTime();
            int next = sampleGreedy(logits, model, sampling, false);
            return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
        }

        float[] logits = forwardPass.get().decode(previousToken, decodeStartPos, model.weights(), config, arch,
                session);
        long forwardNanos = System.nanoTime() - tForward0;
        long tSample0 = System.nanoTime();
        int next = sampleLogits(logits, model, cfg, sampling, false);
        return new StepResult(next, forwardNanos, System.nanoTime() - tSample0);
    }

    private int sampleGreedy(AccelTensor logits, DirectLoadedModel model, SamplingState sampling, boolean firstStep) {
        if (sampling.mode() == SamplingMode.TOKENIZER_AWARE) {
            return sampleGreedyFromTensor(logits, model.config(), model.tokenizer(), model.runtimeTraits(), firstStep,
                    sampling.stops(), sampling.greedyMasks());
        }
        return sampleGreedyFromTensor(logits, model.config());
    }

    private int sampleLogits(float[] logits, DirectLoadedModel model, GenerationConfig cfg, SamplingState sampling,
            boolean firstStep) {
        if (sampling.mode() == SamplingMode.TOKENIZER_AWARE) {
            Tokenizer tokenizer = model.tokenizer();
            return sampleNextToken(logits, tokenizer, cfg, model.config(), model.runtimeTraits(),
                    sampling.frequencies(), sampling.rng(), firstStep, sampling.stops(), tokenSampler.get());
        }
        return tokenSampler.get().sample(logits, cfg, model.config(), sampling.frequencies(), sampling.rng());
    }

    enum SamplingMode {
        RAW_PRETOKENIZED,
        TOKENIZER_AWARE
    }

    record SamplingState(
            SamplingMode mode,
            boolean directGreedy,
            int[] frequencies,
            Random rng,
            Set<Integer> stops,
            GreedySamplingMasks greedyMasks) {
    }

    record StepResult(int token, long forwardNanos, long samplingNanos) {
    }
}
