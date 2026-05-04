package tech.kayys.gollek.plugin;

import tech.kayys.gollek.kernel.OpRegistry;
import tech.kayys.gollek.model.ModelLoader;

public final class PluginContext {
    public final OpRegistry opRegistry;
    public final ModelRegistry modelRegistry;

    public PluginContext(
            OpRegistry opRegistry,
            ModelRegistry modelRegistry) {
        this.opRegistry = opRegistry;
        this.modelRegistry = modelRegistry;
    }
}