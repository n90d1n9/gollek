/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import static tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler.markFirstToken;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.recordSampleFrequency;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Supplier;

final class DirectGenerationLoop {
    private final Supplier<DirectGenerationStepSampler> stepSampler;

    DirectGenerationLoop(Supplier<DirectGenerationStepSampler> stepSampler) {
        this.stepSampler = stepSampler;
    }

    Result run(Request request) {
        StringBuilder out = new StringBuilder();
        List<Long> generatedIds = request.collectTokenIds() ? new ArrayList<>() : Collections.emptyList();
        StreamingDecoder decoder = new StreamingDecoder(request.model().tokenizer(), DecodeOptions.defaultOptions());
        int next = request.firstToken();
        int completionTokens = 0;
        long firstTokenNanos = 0L;
        long decodeNanos = 0L;
        long samplingNanos = 0L;
        int decodeSteps = 0;

        for (int step = 0; step < request.cfg().maxNewTokens(); step++) {
            if (request.stops().contains(next) || request.isCancelled()) {
                break;
            }

            if (firstTokenNanos == 0L) {
                firstTokenNanos = System.nanoTime() - request.requestStartNanos();
            }
            markFirstToken(request.profile(), request.requestStartNanos());
            if (request.collectTokenIds()) {
                generatedIds.add((long) next);
            }
            if (request.countCompletionTokens()) {
                completionTokens++;
            }

            String delta = decoder.decodeNext((long) next);
            if (delta == null && request.normalizeNullDelta()) {
                delta = "";
            }
            out.append(delta);
            if (delta != null && !delta.isEmpty() && request.deltaConsumer() != null) {
                request.deltaConsumer().accept(delta);
            }

            if (endsWithStopString(decoder.currentText(), request.cfg().stopStrings())) {
                break;
            }

            recordSampleFrequency(request.sampling().frequencies(), next);
            if (step + 1 >= request.cfg().maxNewTokens()) {
                break;
            }
            DirectGenerationStepSampler.StepResult decode = stepSampler.get().decode(next,
                    request.decodeStartBase() + step, request.model(), request.session(), request.cfg(),
                    request.sampling());
            next = decode.token();
            decodeNanos += decode.forwardNanos();
            decodeSteps++;
            samplingNanos += decode.samplingNanos();
            if (request.profile() != null) {
                request.profile().decodeNanos += decode.forwardNanos();
                request.profile().decodeSteps++;
                request.profile().samplingNanos += decode.samplingNanos();
            }
            if (request.nextTokenObserver() != null) {
                request.nextTokenObserver().accept(next, step + 1);
            }
        }

        return new Result(out.toString(), generatedIds, completionTokens, firstTokenNanos, decodeNanos,
                samplingNanos, decodeSteps);
    }

    private static boolean endsWithStopString(String text, List<String> stopStrings) {
        if (stopStrings == null || stopStrings.isEmpty()) {
            return false;
        }
        for (String stop : stopStrings) {
            if (text.endsWith(stop)) {
                return true;
            }
        }
        return false;
    }

    record Request(
            DirectLoadedModel model,
            GenerationConfig cfg,
            KVCacheManager.KVCacheSession session,
            DirectGenerationStepSampler.SamplingState sampling,
            Set<Integer> stops,
            int firstToken,
            int decodeStartBase,
            long requestStartNanos,
            InferenceProfile profile,
            boolean collectTokenIds,
            boolean countCompletionTokens,
            boolean normalizeNullDelta,
            BooleanSupplier cancelled,
            Consumer<String> deltaConsumer,
            BiConsumer<Integer, Integer> nextTokenObserver) {
        boolean isCancelled() {
            return cancelled != null && cancelled.getAsBoolean();
        }
    }

    record Result(String text, List<Long> generatedTokenIds, int completionTokens, long firstTokenNanos,
            long decodeNanos, long samplingNanos, int decodeSteps) {
    }
}
