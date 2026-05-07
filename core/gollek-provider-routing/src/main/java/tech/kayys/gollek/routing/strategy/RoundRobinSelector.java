package tech.kayys.gollek.routing.strategy;

import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderSelection;
import tech.kayys.gollek.spi.provider.ProviderSelector;
import tech.kayys.gollek.spi.provider.RoutingContext;
import tech.kayys.gollek.spi.routing.RoutingConfig;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Round-robin selection: cycles through providers sequentially.
 */
public class RoundRobinSelector implements ProviderSelector {

    private final AtomicInteger counter = new AtomicInteger(0);

    @Override
    public ProviderSelection select(
            List<LLMProvider> candidates,
            RoutingContext context,
            RoutingConfig config) {

        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("No candidates available");
        }

        int index = counter.getAndIncrement() % candidates.size();
        LLMProvider selected = candidates.get(index);

        // Build fallback list (remaining providers in order)
        List<LLMProvider> fallbacks = candidates.stream()
                .filter(p -> !p.id().equals(selected.id()))
                .limit(2)
                .toList();

        return ProviderSelection.withFallbacks(selected, fallbacks);
    }

    @Override
    public String description() {
        return "Cycles through providers sequentially";
    }
}
