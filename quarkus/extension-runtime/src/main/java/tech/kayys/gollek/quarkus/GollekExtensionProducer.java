package tech.kayys.gollek.quarkus;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;
import org.eclipse.microprofile.config.inject.ConfigProperty;

@ApplicationScoped
public class GollekExtensionProducer {

    @Produces
    @ApplicationScoped
    public GollekIntegrationService gollekIntegrationService(
            @ConfigProperty(name = "gollek.quarkus.enabled", defaultValue = "false") boolean enabled) {
        if (enabled) {
            System.out.println("[GollekExtension] Quarkus integration enabled at runtime");
            return new RealGollekIntegrationService();
        }
        System.out.println("[GollekExtension] Quarkus integration disabled; using Noop adapters");
        return new NoopGollekIntegrationService();
    }
}

interface GollekIntegrationService {
    String name();
    default boolean enabled() { return false; }
}

class RealGollekIntegrationService implements GollekIntegrationService {
    @Override
    public String name() { return "gollek-quarkus-extension-runtime"; }
    @Override
    public boolean enabled() { return true; }
}

class NoopGollekIntegrationService implements GollekIntegrationService {
    @Override
    public String name() { return "noop-gollek-quarkus-extension"; }
}
