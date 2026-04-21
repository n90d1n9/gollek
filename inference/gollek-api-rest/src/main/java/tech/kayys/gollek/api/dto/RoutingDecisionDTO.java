package tech.kayys.gollek.api.dto;

import tech.kayys.gollek.spi.routing.RoutingDecision;

public record RoutingDecisionDTO(String providerId, String reason) {
    public static RoutingDecisionDTO from(RoutingDecision decision) {
        return new RoutingDecisionDTO(decision.selectedProviderId(), "Score: " + decision.score());
    }
}
