/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.warmup;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.safetensor.engine.planning.InferenceRequestPlanner;
import tech.kayys.gollek.safetensor.engine.planning.PreparedInferenceRequest;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelection;
import tech.kayys.gollek.safetensor.engine.backend.TextExecutionBackendSelector;
import tech.kayys.gollek.safetensor.engine.session.TextExecutionEngineAdminService;
import tech.kayys.gollek.safetensor.engine.session.TextExecutionEngineStatus;
import tech.kayys.gollek.safetensor.engine.session.TextExecutionSessionInspector;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.safetensor.SafetensorProviderConfig;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceEngine;
import tech.kayys.gollek.safetensor.engine.generation.TextInferenceEngine;
import tech.kayys.gollek.safetensor.spi.SafetensorEngine;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderResponse;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.Flow;

/**
 * Direct Safetensor Backend for SafetensorProvider.
 * Delegates requests to the internal native inference engines.
 */
@ApplicationScoped
public class DirectSafetensorBackend {

    private static final Logger log = Logger.getLogger(DirectSafetensorBackend.class);
    private static final String PROVIDER_ID = "safetensor-direct";
    @Inject
    TextInferenceEngine textEngine;

    @Inject
    tech.kayys.gollek.safetensor.audio.SpeechT5Engine speechT5Engine;

    @Inject
    SafetensorProviderConfig config;

    @Inject
    DirectInferenceEngine engine;

    @Inject
    InferenceRequestPlanner requestPlanner;

    @Inject
    TextExecutionBackendSelector backendSelector;

    @Inject
    TextExecutionEngineAdminService textExecutionAdmin;

    private InferenceRequestPlanner planner() {
        return requestPlanner != null ? requestPlanner : new InferenceRequestPlanner(engine, config, selector());
    }

    private TextExecutionBackendSelector selector() {
        return backendSelector != null ? backendSelector : new TextExecutionBackendSelector(engine);
    }

    private TextExecutionEngineAdminService textExecutionAdmin() {
        return textExecutionAdmin != null
                ? textExecutionAdmin
                : new TextExecutionEngineAdminService(new TextExecutionSessionInspector());
    }

    public void initialize(ProviderConfig providerConfig) {
        TextExecutionBackendSelection selection = selector().select();
        log.infof("DirectSafetensorBackend: initialised with FFM memory mapped weights");
        log.infof("DirectSafetensorBackend: text execution backend=%s", selection.selectedBackendId());
        if (selection.fellBack()) {
            log.warnf("DirectSafetensorBackend: %s", selection.fallbackReason());
        }
    }

    public Uni<ProviderHealth> health() {
        TextExecutionEngineStatus status = textExecutionAdmin().status();
        return Uni.createFrom().item(ProviderHealth.builder()
                .status(ProviderHealth.Status.HEALTHY)
                .message("Direct Safetensor backend active (zero-copy enabled)")
                .timestamp(Instant.now())
                .detail("text_execution_active_reusable_sessions", status.activeReusableSessions())
                .detail("text_execution_active_reservations", status.activeReservations())
                .detail("text_execution_resume_capable_sessions", status.resumeCapableSessions())
                .detail("text_execution_backend_session_counts", status.backendSessionCounts())
                .detail("text_execution_reuse_policy_counts", status.reusePolicyCounts())
                .detail("text_execution_artifact_kind_counts", status.artifactKindCounts())
                .build());
    }

    public Uni<InferenceResponse> infer(ProviderRequest request) {
        try {
            PreparedInferenceRequest prepared = planner().prepare(request);
            if (prepared.audioModel()) {
                long synthStart = System.currentTimeMillis();
                int promptTokens = prepared.estimatedPromptTokens();
                java.util.Map<String, Object> metadata = new java.util.HashMap<>();
                metadata.put("execution_backend", prepared.executionBackendId());
                appendPromptMetadata(metadata, prepared);
                appendSessionMetadata(metadata, prepared);
                appendQuantizationMetadata(metadata, prepared.loadedModel());
                metadata.put("audio", java.util.Base64.getEncoder().encodeToString(new byte[0]));
                return speechT5Engine.synthesize(prepared.ttsPrompt(), "alloy", prepared.modelPath(), prepared.audioConfig())
                        .map(bytes -> InferenceResponse.builder()
                                .requestId(request.getRequestId())
                                .model(request.getModel())
                                .content("")
                                .tokensUsed(promptTokens + (bytes.length / 1024))
                                .inputTokens(promptTokens)
                                .outputTokens(bytes.length / 1024)
                                .durationMs(System.currentTimeMillis() - synthStart)
                                .sessionId(request.getSessionId().orElse(null))
                                .metadata(metadata)
                                .metadata("audio", java.util.Base64.getEncoder().encodeToString(bytes))
                                .metadata("modality", "AUDIO")
                                .build());
            }
            return textEngine.generate(prepared.preparedGeneration())
                    .map(infResp -> {
                        InferenceResponse.Builder builder = InferenceResponse.builder()
                            .requestId(infResp.getRequestId())
                            .model(request.getModel())
                            .content(infResp.getContent())
                            .tokensUsed(infResp.getTokensUsed())
                            .inputTokens(infResp.getInputTokens())
                            .outputTokens(infResp.getOutputTokens())
                            .durationMs(infResp.getDurationMs())
                            .finishReason(infResp.getFinishReason())
                            .sessionId(request.getSessionId().orElse(null))
                            .metadata(infResp.getMetadata());
                        builder.metadata("execution_backend", prepared.executionBackendId());
                        appendPromptMetadata(builder, prepared);
                        appendSessionMetadata(builder, prepared);
                        appendQuantizationMetadata(builder, prepared.loadedModel());
                        return builder.build();
                    });
        } catch (Exception e) {
            String msg = (e.getMessage() != null) ? e.getMessage() : "(no message)";
            System.err.println("FATAL in infer [" + e.getClass().getSimpleName() + "]: " + msg);
            e.printStackTrace(System.err);
            log.errorf(e, "Direct inference failed for model %s (resolved path: %s)",
                    request.getModel(), planner().resolveModelPathFromRequest(request));
            return Uni.createFrom().failure(new ProviderException("Direct inference failed for model "
                    + request.getModel() + " [" + e.getClass().getSimpleName() + "]: " + msg, e));
        }
    }

    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        java.util.concurrent.atomic.AtomicInteger idx = new java.util.concurrent.atomic.AtomicInteger(0);
        return Multi.createFrom().publisher(streamToPublisher(request))
                .map(response -> {
                    // DirectInferenceEngine leaves durationMs=0 on per-token deltas; the summary frame sets it.
                    boolean isFinal = response.getDurationMs() > 0L;
                    int index = idx.getAndIncrement();
                    Map<String, Object> metadata = response.getMetadata();
                    if ("AUDIO".equals(response.getMetadata().get("modality"))) {
                        return new StreamingInferenceChunk(response.getRequestId(), index, 
                                tech.kayys.gollek.spi.model.ModalityType.AUDIO, 
                                response.getContent(), null, isFinal, response.getFinishReason(), null, Instant.now(), metadata,
                                null, null, null);
                    }
                    if (isFinal) {
                        return new StreamingInferenceChunk(response.getRequestId(), index,
                                tech.kayys.gollek.spi.model.ModalityType.TEXT,
                                response.getContent(), null, true, response.getFinishReason(), null, Instant.now(), metadata,
                                null, null, null);
                    } else {
                        return StreamingInferenceChunk.withMetadata(
                                response.getRequestId(), index, response.getContent(), metadata);
                    }
                });
    }

    private Flow.Publisher<ProviderResponse> streamToPublisher(ProviderRequest request) {
        try {
            PreparedInferenceRequest prepared = planner().prepare(request);
            if (prepared.audioModel()) {
                long synthStart = System.currentTimeMillis();
                int promptTokens = prepared.estimatedPromptTokens();
                return Multi.createFrom().publisher(
                        speechT5Engine.synthesize(prepared.ttsPrompt(), "alloy", prepared.modelPath(), prepared.audioConfig())
                            .map(bytes -> {
                                long durationMs = System.currentTimeMillis() - synthStart;
                                String b64 = java.util.Base64.getEncoder().encodeToString(bytes);
                                return ProviderResponse.builder()
                                        .requestId(request.getRequestId())
                                        .model(request.getModel())
                                        .content(b64)
                                        .finishReason("stop")
                                        .promptTokens(promptTokens)
                                        .completionTokens(bytes.length / 1024) // rough "token" equivalent for audio bytes
                                        .metadata("modality", "AUDIO")
                                        .durationMs(durationMs)
                                        .build();
                            })
                            .toMulti()
                );
            }

            // Generate response lazily
            return textEngine.generateStream(prepared.preparedGeneration())
                    .map(infResp -> ProviderResponse.builder()
                            .requestId(infResp.getRequestId())
                            .content(infResp.getContent())
                            .model(infResp.getModel())
                            .finishReason(
                                    infResp.getFinishReason() != null ? infResp.getFinishReason().name().toLowerCase()
                                            : "stop")
                            .promptTokens(infResp.getInputTokens())
                            .completionTokens(infResp.getOutputTokens())
                            .totalTokens(infResp.getTokensUsed())
                            .durationMs(infResp.getDurationMs())
                            .metadata(infResp.getMetadata())
                            .build());
        } catch (Exception e) {
            String msg = (e.getMessage() != null) ? e.getMessage() : "(no message)";
            System.err.println("FATAL in streamToPublisher [" + e.getClass().getSimpleName() + "]: " + msg);
            e.printStackTrace(System.err);
            log.errorf(e, "Direct inference failed for model %s (resolved path: %s)",
                    request.getModel(), planner().resolveModelPathFromRequest(request));
            throw new ProviderException("Direct inference failed for model "
                    + request.getModel() + " [" + e.getClass().getSimpleName() + "]: " + msg, e);
        }
    }

    private void appendQuantizationMetadata(InferenceResponse.Builder builder, SafetensorEngine.LoadedModel loadedModel) {
        if (loadedModel == null || !loadedModel.isQuantized()) {
            return;
        }
        if (loadedModel instanceof DirectInferenceEngine.LoadedModel directLoadedModel) {
            if (directLoadedModel.getQuantCacheState() != null) {
                builder.metadata("quant_cache_state", directLoadedModel.getQuantCacheState());
            }
            if (directLoadedModel.getQuantCachePath() != null) {
                builder.metadata("quant_cache_path", directLoadedModel.getQuantCachePath().toString());
            }
            builder.metadata("quant_strategy", directLoadedModel.getQuantStrategy().name().toLowerCase());
        }
    }

    private void appendQuantizationMetadata(java.util.Map<String, Object> metadata, SafetensorEngine.LoadedModel loadedModel) {
        if (loadedModel == null || !loadedModel.isQuantized()) {
            return;
        }
        if (loadedModel instanceof DirectInferenceEngine.LoadedModel directLoadedModel) {
            if (directLoadedModel.getQuantCacheState() != null) {
                metadata.put("quant_cache_state", directLoadedModel.getQuantCacheState());
            }
            if (directLoadedModel.getQuantCachePath() != null) {
                metadata.put("quant_cache_path", directLoadedModel.getQuantCachePath().toString());
            }
            metadata.put("quant_strategy", directLoadedModel.getQuantStrategy().name().toLowerCase());
        }
    }

    private void appendPromptMetadata(InferenceResponse.Builder builder, PreparedInferenceRequest prepared) {
        if (prepared.preparedPrompt() == null) {
            return;
        }
        builder.metadata("prompt_fingerprint", prepared.preparedPrompt().fingerprint());
        builder.metadata("prompt_reuse_strategy", prepared.promptReusePlan().strategy().name().toLowerCase());
        if (prepared.promptReusePlan().reuseKey() != null) {
            builder.metadata("prompt_reuse_key", prepared.promptReusePlan().reuseKey());
        }
    }

    private void appendPromptMetadata(java.util.Map<String, Object> metadata, PreparedInferenceRequest prepared) {
        if (prepared.preparedPrompt() == null) {
            return;
        }
        metadata.put("prompt_fingerprint", prepared.preparedPrompt().fingerprint());
        metadata.put("prompt_reuse_strategy", prepared.promptReusePlan().strategy().name().toLowerCase());
        if (prepared.promptReusePlan().reuseKey() != null) {
            metadata.put("prompt_reuse_key", prepared.promptReusePlan().reuseKey());
        }
    }

    private void appendSessionMetadata(InferenceResponse.Builder builder, PreparedInferenceRequest prepared) {
        if (prepared.sessionPlan() == null) {
            return;
        }
        builder.metadata("session_reuse_decision", prepared.sessionReuseDecision().status().name().toLowerCase());
        builder.metadata("session_prefill_strategy", prepared.sessionPlan().prefillStrategy().name().toLowerCase());
        builder.metadata("session_decode_strategy", prepared.sessionPlan().decodeStrategy().name().toLowerCase());
        builder.metadata("session_reuse_policy", prepared.sessionPlan().reusePolicy().name().toLowerCase());
        builder.metadata("session_scope", prepared.sessionPlan().scope().name().toLowerCase());
        if (prepared.preparationPlan() != null) {
            builder.metadata("session_preparation_rationale", prepared.preparationPlan().rationale());
        }
        if (prepared.preparationContext() != null) {
            builder.metadata("session_preparation_backend", prepared.preparationContext().backendId());
            builder.metadata("session_preparation_model_family", prepared.preparationContext().modelFamily());
            builder.metadata("session_preparation_architecture", prepared.preparationContext().primaryArchitecture());
            if (prepared.preparationContext().conversationSessionId() != null) {
                builder.metadata("conversation_session_id", prepared.preparationContext().conversationSessionId());
            }
            builder.metadata("session_preparation_hardware", prepared.preparationContext().hardwareProfile().name().toLowerCase());
            builder.metadata("session_preparation_profile", prepared.preparationContext().planningProfile().name().toLowerCase());
        }
        if (prepared.conversationExecutionState() != null) {
            builder.metadata("conversation_session_status", prepared.conversationExecutionState().status().name().toLowerCase());
            builder.metadata("conversation_session_rationale", prepared.conversationExecutionState().rationale());
            if (prepared.conversationExecutionState().requestedModelKey() != null) {
                builder.metadata("conversation_session_model_key", prepared.conversationExecutionState().requestedModelKey());
            }
            if (prepared.conversationExecutionState().existingModelKey() != null) {
                builder.metadata("conversation_session_existing_model_key", prepared.conversationExecutionState().existingModelKey());
            }
            builder.metadata("conversation_session_cached_tokens", prepared.conversationExecutionState().cachedTokens());
            builder.metadata("conversation_session_kv_state_available", prepared.conversationExecutionState().kvStateAvailable());
            builder.metadata("conversation_session_pending_replay_available",
                    prepared.conversationExecutionState().pendingReplayTokenId() != null);
        }
        if (prepared.artifactPlan() != null) {
            builder.metadata("session_artifact_strategy", prepared.artifactPlan().strategy().name().toLowerCase());
        }
        if (prepared.sessionPlan().sessionKey() != null) {
            builder.metadata("session_key", prepared.sessionPlan().sessionKey());
        }
        if (prepared.sessionReuseDecision().sessionKey() != null) {
            builder.metadata("session_reuse_candidate_key", prepared.sessionReuseDecision().sessionKey());
        }
        if (prepared.sessionDescriptor() != null) {
            builder.metadata("session_descriptor_key", prepared.sessionDescriptor().sessionKey());
            builder.metadata("session_descriptor_prompt_fingerprint", prepared.sessionDescriptor().promptFingerprint());
        }
    }

    private void appendSessionMetadata(java.util.Map<String, Object> metadata, PreparedInferenceRequest prepared) {
        if (prepared.sessionPlan() == null) {
            return;
        }
        metadata.put("session_reuse_decision", prepared.sessionReuseDecision().status().name().toLowerCase());
        metadata.put("session_prefill_strategy", prepared.sessionPlan().prefillStrategy().name().toLowerCase());
        metadata.put("session_decode_strategy", prepared.sessionPlan().decodeStrategy().name().toLowerCase());
        metadata.put("session_reuse_policy", prepared.sessionPlan().reusePolicy().name().toLowerCase());
        metadata.put("session_scope", prepared.sessionPlan().scope().name().toLowerCase());
        if (prepared.preparationPlan() != null) {
            metadata.put("session_preparation_rationale", prepared.preparationPlan().rationale());
        }
        if (prepared.preparationContext() != null) {
            metadata.put("session_preparation_backend", prepared.preparationContext().backendId());
            metadata.put("session_preparation_model_family", prepared.preparationContext().modelFamily());
            metadata.put("session_preparation_architecture", prepared.preparationContext().primaryArchitecture());
            if (prepared.preparationContext().conversationSessionId() != null) {
                metadata.put("conversation_session_id", prepared.preparationContext().conversationSessionId());
            }
            metadata.put("session_preparation_hardware", prepared.preparationContext().hardwareProfile().name().toLowerCase());
            metadata.put("session_preparation_profile", prepared.preparationContext().planningProfile().name().toLowerCase());
        }
        if (prepared.conversationExecutionState() != null) {
            metadata.put("conversation_session_status", prepared.conversationExecutionState().status().name().toLowerCase());
            metadata.put("conversation_session_rationale", prepared.conversationExecutionState().rationale());
            if (prepared.conversationExecutionState().requestedModelKey() != null) {
                metadata.put("conversation_session_model_key", prepared.conversationExecutionState().requestedModelKey());
            }
            if (prepared.conversationExecutionState().existingModelKey() != null) {
                metadata.put("conversation_session_existing_model_key", prepared.conversationExecutionState().existingModelKey());
            }
            metadata.put("conversation_session_cached_tokens", prepared.conversationExecutionState().cachedTokens());
            metadata.put("conversation_session_kv_state_available", prepared.conversationExecutionState().kvStateAvailable());
            metadata.put("conversation_session_pending_replay_available",
                    prepared.conversationExecutionState().pendingReplayTokenId() != null);
        }
        if (prepared.artifactPlan() != null) {
            metadata.put("session_artifact_strategy", prepared.artifactPlan().strategy().name().toLowerCase());
        }
        if (prepared.sessionPlan().sessionKey() != null) {
            metadata.put("session_key", prepared.sessionPlan().sessionKey());
        }
        if (prepared.sessionReuseDecision().sessionKey() != null) {
            metadata.put("session_reuse_candidate_key", prepared.sessionReuseDecision().sessionKey());
        }
        if (prepared.sessionDescriptor() != null) {
            metadata.put("session_descriptor_key", prepared.sessionDescriptor().sessionKey());
            metadata.put("session_descriptor_prompt_fingerprint", prepared.sessionDescriptor().promptFingerprint());
        }
    }

}
