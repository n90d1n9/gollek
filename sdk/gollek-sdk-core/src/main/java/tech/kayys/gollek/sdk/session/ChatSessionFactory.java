package tech.kayys.gollek.sdk.session;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.sdk.core.GollekSdk;

/**
 * Factory for creating ChatSession instances.
 */
@ApplicationScoped
public class ChatSessionFactory {

    @Inject
    GollekSdk sdk;

    public ChatSession createSession(String modelId, String providerId) {
        return createSession(modelId, providerId, true);
    }

    public ChatSession createSession(String modelId, String providerId, boolean enableSession) {
        return new ChatSessionImpl(sdk, modelId, providerId, enableSession);
    }
}
