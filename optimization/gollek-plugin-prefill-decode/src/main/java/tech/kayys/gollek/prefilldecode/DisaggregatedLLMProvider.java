package tech.kayys.gollek.prefilldecode;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.inference.InferenceRequest;

import java.util.Set;

/**
 * A provider that implements disaggregated prefill/decode logic.
 * It uses the PrefillDecodeDisaggService to orchestrate the handoff.
 */
@ApplicationScoped
public class DisaggregatedLLMProvider implements StreamingProvider {

    private static final Logger LOG = Logger.getLogger(DisaggregatedLLMProvider.class);
    public static final String ID = "pd-disagg";

    @Inject
    PrefillDecodeDisaggService disaggService;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public String name() {
        return "Disaggregated Prefill-Decode Provider";
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .supportedFormats(Set.of()) // Internal provider
                .supportedDevices(Set.of())
                .build();
    }

    @Override
    public boolean supports(String model, ProviderRequest request) {
        // Active when explicitly requested via parameters or a specific routing hint
        if (request.getParameters() != null && Boolean.TRUE.equals(request.getParameters().get("disaggregated"))) {
            return true;
        }
        return model.endsWith("-pd") || model.endsWith("-disagg");
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Multi.createBy().concatenating().streams(
                inferStream(request)
        ).collect().asList().map(chunks -> {
            StringBuilder sb = new StringBuilder();
            for (StreamingInferenceChunk chunk : chunks) {
                if (chunk.delta() != null) {
                    sb.append(chunk.delta());
                }
            }
            return InferenceResponse.builder()
                    .requestId(request.getRequestId())
                    .content(sb.toString())
                    .model(request.getModel())
                    .build();
        });
    }

    @Override
    public void initialize(ProviderConfig config) {
        try {
            String backend = config != null ? config.getString("kv-transfer-backend", null) : null;
            boolean enable = config != null && config.getBoolean("enabled", false);
            disaggService.configure(backend, enable);
            LOG.infof("[DisaggProvider] Initialized — backend=%s, enable=%s", backend, enable);
        } catch (Exception e) {
            LOG.errorf("[DisaggProvider] Initialization failed: %s", e.getMessage());
        }
    }

    @Override
    public void shutdown() {
        try {
            disaggService.stop();
            LOG.info("[DisaggProvider] Shutdown complete");
        } catch (Exception e) {
            LOG.errorf("[DisaggProvider] Shutdown error: %s", e.getMessage());
        }
    }

    @Override
    public Uni<ProviderHealth> health() {
        boolean active = disaggService.isActive();
        return Uni.createFrom().item(
            active ? ProviderHealth.healthy("PD service active")
                   : ProviderHealth.unhealthy("Disaggregation service inactive")
        );
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(ID)
                .name("Disaggregated Prefill-Decode Provider")
                .version("1.0.0")
                .description("In-process disaggregated prefill/decode via PagedAttention FFM kernels. " +
                        "Supports IPC and NIXL DMA KV transfer backends.")
                .vendor("Kayys.tech")
                .build();
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        InferenceRequest infReq = toInferenceRequest(request);
        
        // 1. Trigger Prefill
        return disaggService.executePrefillAsync(infReq)
                .onItem().transformToMulti(kvTransferId -> {
                    // 2. Trigger Decode Stream using the handoff token
                    return disaggService.executeDecodeStream(kvTransferId, infReq);
                });
    }

    private InferenceRequest toInferenceRequest(ProviderRequest request) {
        return InferenceRequest.builder()
                .requestId(request.getRequestId())
                .model(request.getModel())
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .streaming(request.isStreaming())
                .build();
    }
}
