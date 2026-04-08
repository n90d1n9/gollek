package tech.kayys.gollek.runtime.unified.impl;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Default;
import jakarta.ws.rs.container.ContainerRequestContext;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.context.RequestContextResolver;
import java.util.UUID;

@ApplicationScoped
@Default
public class DefaultRequestContextResolver implements RequestContextResolver {

    @Override
    public RequestContext resolve(ContainerRequestContext request) {
        // Return a default tenant context for the unified runtime
        String reqId = request.getHeaderString("X-Request-Id");
        if (reqId == null || reqId.isEmpty()) {
            reqId = UUID.randomUUID().toString();
        }
        return RequestContextResolver.DefaultRequestContext.anonymous(reqId);
    }
}