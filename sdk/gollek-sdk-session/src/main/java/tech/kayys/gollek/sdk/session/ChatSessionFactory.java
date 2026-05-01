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
        return new ChatSessionImpl(sdk, modelId, providerId);
    }
}
