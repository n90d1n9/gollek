package tech.kayys.gollek.cli.util;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;
import tech.kayys.gollek.plugin.runner.gguf.GgufRunnerPlugin;

import static org.junit.jupiter.api.Assertions.assertTrue;

class PluginAvailabilityCheckerTest {
    @Test
    void cliClasspathIncludesServiceLoadedGgufRunnerPlugin() {
        assertTrue(RunnerPluginManager.getInstance().getPlugin(GgufRunnerPlugin.ID).isPresent());
    }

    @Test
    void availabilityCheckerIncludesServiceLoadedRunnerPlugins() {
        PluginAvailabilityChecker checker = new PluginAvailabilityChecker();

        assertTrue(checker.hasRunnerPlugins());
        assertTrue(checker.getRunnerPluginIds().contains(GgufRunnerPlugin.ID));
    }
}
