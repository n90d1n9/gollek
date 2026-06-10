package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentServingPreflightPolicyTest {

    @Test
    void exposesNamedProfilesAsMetadata() {
        AgentServingPreflightPolicy readinessOnly = AgentServingPreflightPolicy.profile("readiness-only");
        AgentServingPreflightPolicy requestedRoute = AgentServingPreflightPolicy.profile("requested_route");
        AgentServingPreflightPolicy clean = AgentServingPreflightPolicy.profile("ci");

        assertEquals(
                java.util.List.of(
                        AgentServingPreflightPolicy.PROFILE_READINESS_ONLY,
                        AgentServingPreflightPolicy.PROFILE_REQUESTED_ROUTE,
                        AgentServingPreflightPolicy.PROFILE_CLEAN),
                AgentServingPreflightPolicy.supportedProfileNames());
        assertTrue(AgentServingPreflightPolicy.supportsProfile("strict-route"));
        assertTrue(AgentServingPreflightPolicy.supportsProfile("ci"));
        assertFalse(AgentServingPreflightPolicy.supportsProfile("cleen"));
        assertEquals(AgentServingPreflightPolicy.PROFILE_READINESS_ONLY, readinessOnly.profile());
        assertFalse(readinessOnly.requireRouteMatch());
        assertFalse(readinessOnly.failOnWarnings());
        assertEquals(AgentServingPreflightPolicy.PROFILE_REQUESTED_ROUTE, requestedRoute.profile());
        assertTrue(requestedRoute.requireRouteMatch());
        assertFalse(requestedRoute.failOnWarnings());
        assertEquals(AgentServingPreflightPolicy.PROFILE_CLEAN, clean.profile());
        assertTrue(clean.requireRouteMatch());
        assertTrue(clean.failOnWarnings());
        assertEquals(Map.of(
                        "object", AgentServingPreflightPolicy.OBJECT,
                        "profile", AgentServingPreflightPolicy.PROFILE_CLEAN,
                        "route_match_required", true,
                        "warnings_blocking", true),
                clean.toMetadata());
    }

    @Test
    void derivesCustomProfileFromBooleanOptions() {
        AgentServingPreflightPolicy warningsOnly = AgentServingPreflightPolicy.of(false, true);

        assertEquals(AgentServingPreflightPolicy.PROFILE_CUSTOM, warningsOnly.profile());
        assertFalse(warningsOnly.requireRouteMatch());
        assertTrue(warningsOnly.failOnWarnings());
    }

    @Test
    void requireProfileRejectsUnknownNames() {
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> AgentServingPreflightPolicy.requireProfile("cleen"));

        assertTrue(exception.getMessage().contains("Unsupported preflight policy profile: cleen"));
        assertTrue(exception.getMessage().contains("readiness_only"));
        assertTrue(exception.getMessage().contains("requested_route"));
        assertTrue(exception.getMessage().contains("clean"));
    }

    @Test
    void readsProfileAndOverridesFromMetadata() {
        AgentServingPreflightPolicy policy = AgentServingPreflightPolicy.fromMetadata(Map.of(
                "profile", "requested_route",
                "warnings_blocking", true));

        assertEquals(AgentServingPreflightPolicy.PROFILE_CLEAN, policy.profile());
        assertTrue(policy.requireRouteMatch());
        assertTrue(policy.failOnWarnings());
    }
}
