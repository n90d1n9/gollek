/*
 * Copyright (c) 2026 Kayys.tech
 */
package tech.kayys.gollek.routing.intelligent;

import tech.kayys.gollek.spi.runtime.Capability;
import tech.kayys.gollek.spi.runtime.CapabilityProfile;

import java.util.Set;

/**
 * Validates whether an ExecutionProvider satisfies the capabilities
 * requested by a given RoutingContext.
 */
public final class CapabilityMatcher {

    private CapabilityMatcher() {
        // utility class
    }

    /**
     * Checks if a provider's capability profile satisfies all required capabilities.
     *
     * @param providerCapabilities the capabilities the provider claims
     * @param required             the capabilities strictly required by the request
     * @return true if all required capabilities are supported, false otherwise
     */
    public static boolean supports(
            CapabilityProfile providerCapabilities,
            Set<Capability> required
    ) {
        if (required == null || required.isEmpty()) {
            return true;
        }
        return providerCapabilities.supported().containsAll(required);
    }
}
