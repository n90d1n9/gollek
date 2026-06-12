package tech.kayys.gollek.engine.inference;

import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.inference.InferencePipeline;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.plugin.cache.SemanticCachePlugin;
import tech.kayys.gollek.spi.exception.PluginException;

import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.InferencePhase;

import java.util.ArrayList;
import java.util.List;
import java.util.Comparator;

/**
 * Default implementation of the inference pipeline.
 * Manages and executes phase plugins, including semantic caching.
 */
public class DefaultInferencePipeline implements InferencePipeline {

    private final List<InferencePhasePlugin> plugins = new ArrayList<>();
    private final EngineContext engineContext;

    public DefaultInferencePipeline(EngineContext engineContext) {
        this.engineContext = engineContext;
        registerDefaultPlugins();
    }

    private void registerDefaultPlugins() {
        // Register semantic cache plugin
        registerPlugin(new SemanticCachePlugin());
    }

    public void registerPlugin(InferencePhasePlugin plugin) {
        plugins.add(plugin);
        plugins.sort(Comparator.comparingInt(InferencePhasePlugin::order));
    }

    @Override
    public Uni<ExecutionContext> execute(ExecutionContext context) {
        Uni<ExecutionContext> pipeline = Uni.createFrom().item(context);
        for (InferencePhase phase : InferencePhase.values()) {
            pipeline = pipeline.chain(ctx -> executePhase(ctx, phase));
        }
        return pipeline;
    }

    @Override
    public Uni<ExecutionContext> executePhase(ExecutionContext context, InferencePhase phase) {
        try {
            for (InferencePhasePlugin plugin : plugins) {
                if (plugin.phase() == phase && plugin.shouldExecute(context)) {
                    plugin.execute(context, engineContext);
                    
                    // Check if a plugin requested a short circuit (e.g. cache hit)
                    Boolean shortCircuit = (Boolean) context.metadata().get("shortCircuit");
                    if (Boolean.TRUE.equals(shortCircuit)) {
                        break;
                    }
                }
            }
            return Uni.createFrom().item(context);
        } catch (Exception e) {
            return Uni.createFrom().failure(e);
        }
    }
}
