package tech.kayys.gollek.plugin;

import tech.kayys.gollek.ir.OpRegistry;
import tech.kayys.gollek.ir.ModelRegistry;

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