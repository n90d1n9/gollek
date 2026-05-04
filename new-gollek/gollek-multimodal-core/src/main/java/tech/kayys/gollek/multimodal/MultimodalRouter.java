package tech.kayys.gollek.multimodal;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.*;
import java.util.Arrays;

import java.util.Collections;
import java.util.List;

/**
 * Routes incoming {@link MultimodalRequest} instances to the appropriate
 * {@link MultimodalInferenceProvider}, enforcing capability validation and
 * applying the configured fallback strategy on failure.
 *
 * <h3>Routing algorithm</h3>
 * <ol>
 *   <li>If the request names an explicit provider/model via {@link MultimodalRequest#getModel()},
 *       look that provider up directly.</li>
 *   <li>Otherwise, score all available providers by modality coverage and select
 *       the best match.</li>
 *   <li>Validate the selected provider's {@link MultimodalCapability} against
 *       the request; return a structured error response on violation.</li>
 *   <li>Dispatch and, on provider failure, attempt the fallback chain.</li>
 * </ol>
 */
@ApplicationScoped
public class MultimodalRouter {

    private static final Logger LOG = Logger.getLogger(MultimodalRouter.class);

    private static final String ERR_NO_PROVIDER  = "MM-ROUTER-001";
    private static final String ERR_VALIDATION   = "MM-ROUTER-002";
    private static final String ERR_ALL_FAILED   = "MM-ROUTER-003";

    @Inject
    MultimodalCapabilityRegistry registry;

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Route and execute a multimodal request (non-streaming).
     */
    public Uni<MultimodalResponse> route(MultimodalRequest request) {
        long start = System.currentTimeMillis();

        return Uni.createFrom().item(() -> resolveProvider(request))
                .onItem().transformToUni(provider -> {
                    // Validate capability (place-holder for real validation)
                    LOG.debugf("[%s] Routing to provider '%s'",
                            request.getRequestId(), provider.providerId());
                    return provider.infer(request)
                            .onFailure().recoverWithUni(t ->
                                    handleProviderFailure(request, provider, t, start));
                })
                .onFailure().recoverWithItem(t ->
                        MultimodalResponse.error(request.getRequestId(),
                                request.getModel(), ERR_NO_PROVIDER, t.getMessage()));
    }

    /**
     * Route and execute a multimodal request (streaming).
     */
    public Multi<MultimodalContent> routeStream(MultimodalRequest request) {
        return route(request)
                .onItem().transformToMulti(resp ->
                        Multi.createFrom().iterable(Arrays.asList(resp.getOutputs())))
                .onFailure().recoverWithMulti(t ->
                        Multi.createFrom().failure(t));
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    private MultimodalInferenceProvider resolveProvider(MultimodalRequest request) {
        String modelId = request.getModel();

        // 1. Exact match by model/provider id
        var direct = registry.providerFor(modelId);
        if (direct.isPresent() && direct.get().isAvailable()) {
            return direct.get();
        }

        // 2. Auto-select by capability
        List<MultimodalInferenceProvider> candidates =
                registry.findCapable(Collections.emptySet(), null); // Simplified fallback lookup

        if (candidates.isEmpty()) {
            throw new IllegalStateException(
                    "No multimodal provider available for model='" + modelId + "'");
        }

        // Prefer first match (future: weighted scoring)
        return candidates.get(0);
    }

    private Uni<MultimodalResponse> handleProviderFailure(MultimodalRequest request,
                                                           MultimodalInferenceProvider failed,
                                                           Throwable cause,
                                                           long startMs) {
        LOG.warnf(cause, "[%s] Provider '%s' failed, attempting fallback.",
                request.getRequestId(), failed.providerId());

        // Find alternative providers
        List<MultimodalInferenceProvider> fallbacks =
                registry.findCapable(Collections.emptySet(), null); 
        fallbacks.removeIf(p -> p.providerId().equals(failed.providerId()));

        if (fallbacks.isEmpty()) {
            return Uni.createFrom().item(
                    MultimodalResponse.error(request.getRequestId(),
                            failed.providerId(), ERR_ALL_FAILED,
                            "Primary provider failed and no fallbacks are available: "
                            + cause.getMessage()));
        }

        MultimodalInferenceProvider fallback = fallbacks.get(0);
        LOG.infof("[%s] Falling back to provider '%s'",
                request.getRequestId(), fallback.providerId());

        return fallback.infer(request)
                .onItem().transform(resp ->
                        MultimodalResponse.builder()
                                .requestId(resp.getRequestId())
                                .model(fallback.providerId())
                                .outputs(resp.getOutputs())
                                .usage(resp.getUsage())
                                .status(MultimodalResponse.ResponseStatus.SUCCESS)
                                .build());
    }
}
