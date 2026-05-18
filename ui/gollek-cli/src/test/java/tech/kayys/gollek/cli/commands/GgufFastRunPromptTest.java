package tech.kayys.gollek.cli.commands;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufFastRunPromptTest {

    @Test
    void wrapsGemma4PromptWithTurnTokens() {
        String formatted = GgufFastRun.formatPromptForModel(
                "where is jakarta",
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));

        assertEquals("<|turn>user\nAnswer directly and concisely.\nQuestion: where is jakarta<turn|>\n<|turn>model\n", formatted);
    }

    @Test
    void keepsGemma4NonQuestionPromptWithoutConciseQuestionInstruction() {
        String formatted = GgufFastRun.formatPromptForModel(
                "write a tiny poem about jakarta",
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));

        assertEquals("<|turn>user\nwrite a tiny poem about jakarta<turn|>\n<|turn>model\n", formatted);
    }

    @Test
    void conciseQuestionInstructionCanBeDisabled() {
        String previous = System.getProperty("gollek.gguf.fast_run.concise_qa_prompt");
        System.setProperty("gollek.gguf.fast_run.concise_qa_prompt", "false");
        try {
            String formatted = GgufFastRun.formatPromptForModel(
                    "where is jakarta",
                    Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf"));

            assertEquals("<|turn>user\nwhere is jakarta<turn|>\n<|turn>model\n", formatted);
        } finally {
            restoreProperty("gollek.gguf.fast_run.concise_qa_prompt", previous);
        }
    }

    @Test
    void keepsPreformattedPromptUntouched() {
        String prompt = "<|turn>user\nwhere is jakarta<turn|>\n<|turn>model\n";

        assertEquals(prompt, GgufFastRun.formatPromptForModel(
                prompt,
                Path.of("/tmp/google__gemma-4-E2B-it-Q4_K_M.gguf")));
    }

    @Test
    void keepsNonGemma4PromptRawInAutoMode() {
        assertEquals("where is jakarta", GgufFastRun.formatPromptForModel(
                "where is jakarta",
                Path.of("/tmp/llama.gguf")));
    }

    @Test
    void providerJavaSelectsJavaEngineByDefault() {
        assertEquals("JAVA", GgufFastRun.effectiveEngineModeName("java", "auto"));
    }

    @Test
    void explicitEngineOverridesProviderDefault() {
        assertEquals("LLAMA_CPP", GgufFastRun.effectiveEngineModeName("java", "llama.cpp"));
    }

    @Test
    void benchmarkAliasSelectsBenchmarkMode() {
        assertEquals("BENCHMARK", GgufFastRun.effectiveEngineModeName("gguf", "compare"));
    }

    @Test
    void javaReadinessSummaryIsMachineReadable() {
        GgufRuntimeProfile profile = new GgufRuntimeProfile(
                "gemma4",
                3,
                1,
                1,
                1024,
                1024,
                1024,
                2,
                2,
                0,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                new ModelConfig(),
                1);

        assertEquals(
                "loaderReady=true, decoderTensorsReady=true, rowDotReady=true, generationReady=false",
                GgufFastRun.javaReadinessSummary(profile));
    }

    @Test
    void boundsBatchConfigurationToContextWindow() {
        String previous = System.getProperty("gollek.gguf.fast_run.batch");
        System.setProperty("gollek.gguf.fast_run.batch", "4096");
        try {
            assertEquals(2048, GgufFastRun.boundedBatch("gollek.gguf.fast_run.batch", 1024, 2048));
        } finally {
            restoreProperty("gollek.gguf.fast_run.batch", previous);
        }
    }

    @Test
    void keepsBatchConfigurationPositive() {
        String previous = System.getProperty("gollek.gguf.fast_run.ubatch");
        System.setProperty("gollek.gguf.fast_run.ubatch", "0");
        try {
            assertEquals(1, GgufFastRun.boundedBatch("gollek.gguf.fast_run.ubatch", 512, 1024));
        } finally {
            restoreProperty("gollek.gguf.fast_run.ubatch", previous);
        }
    }

    @Test
    void autoContextUsesSmallWindowForShortPrompt() {
        String previous = System.getProperty("gollek.gguf.fast_run.context");
        System.clearProperty("gollek.gguf.fast_run.context");
        try {
            assertEquals(512, GgufFastRun.fastRunContext("where is jakarta", 24));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previous);
        }
    }

    @Test
    void autoContextGrowsForLongPrompt() {
        String previous = System.getProperty("gollek.gguf.fast_run.context");
        System.clearProperty("gollek.gguf.fast_run.context");
        try {
            assertEquals(2048, GgufFastRun.fastRunContext("x".repeat(1000), 24));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previous);
        }
    }

    @Test
    void explicitContextOverridesAutoSizing() {
        String previous = System.getProperty("gollek.gguf.fast_run.context");
        System.setProperty("gollek.gguf.fast_run.context", "1536");
        try {
            assertEquals(1536, GgufFastRun.fastRunContext("where is jakarta", 24));
        } finally {
            restoreProperty("gollek.gguf.fast_run.context", previous);
        }
    }

    @Test
    void fastRunSwaFullDefaultsToLlamaCppCommonMode() {
        String previous = System.getProperty("gollek.gguf.fast_run.swa_full");
        System.clearProperty("gollek.gguf.fast_run.swa_full");
        try {
            assertFalse(GgufFastRun.fastRunSwaFull());
        } finally {
            restoreProperty("gollek.gguf.fast_run.swa_full", previous);
        }
    }

    @Test
    void fastRunSwaFullCanBeEnabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.swa_full");
        System.setProperty("gollek.gguf.fast_run.swa_full", "true");
        try {
            assertTrue(GgufFastRun.fastRunSwaFull());
        } finally {
            restoreProperty("gollek.gguf.fast_run.swa_full", previous);
        }
    }

    @Test
    void fastRunCpuFallbackIsEnabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.cpu_fallback");
        System.clearProperty("gollek.gguf.fast_run.cpu_fallback");
        try {
            assertTrue(GgufFastRun.fastRunCpuFallback());
        } finally {
            restoreProperty("gollek.gguf.fast_run.cpu_fallback", previous);
        }
    }

    @Test
    void fastRunCpuFallbackCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.cpu_fallback");
        System.setProperty("gollek.gguf.fast_run.cpu_fallback", "false");
        try {
            assertFalse(GgufFastRun.fastRunCpuFallback());
        } finally {
            restoreProperty("gollek.gguf.fast_run.cpu_fallback", previous);
        }
    }

    @Test
    void fastRunTimingIsDisabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.timing");
        System.clearProperty("gollek.gguf.fast_run.timing");
        try {
            assertFalse(GgufFastRun.fastRunTiming());
        } finally {
            restoreProperty("gollek.gguf.fast_run.timing", previous);
        }
    }

    @Test
    void fastRunTimingCanBeEnabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.timing");
        System.setProperty("gollek.gguf.fast_run.timing", "true");
        try {
            assertTrue(GgufFastRun.fastRunTiming());
        } finally {
            restoreProperty("gollek.gguf.fast_run.timing", previous);
        }
    }

    @Test
    void oneShotNativeRunUsesHardExitByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.clearProperty("gollek.gguf.fast_run.hard_exit_after_run");
        try {
            assertTrue(GgufFastRun.shouldHardExitAfterNativeRun(false));
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void daemonNativeRunDoesNotHardExitAfterRequest() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.clearProperty("gollek.gguf.fast_run.hard_exit_after_run");
        try {
            assertFalse(GgufFastRun.shouldHardExitAfterNativeRun(true));
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void oneShotNativeHardExitCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.setProperty("gollek.gguf.fast_run.hard_exit_after_run", "false");
        try {
            assertFalse(GgufFastRun.shouldHardExitAfterNativeRun(false));
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void staleDaemonForceKillIsEnabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.stale_daemon_force_kill");
        System.clearProperty("gollek.gguf.fast_run.stale_daemon_force_kill");
        try {
            assertTrue(GgufFastRun.staleDaemonForceKill());
        } finally {
            restoreProperty("gollek.gguf.fast_run.stale_daemon_force_kill", previous);
        }
    }

    @Test
    void staleDaemonForceKillCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.stale_daemon_force_kill");
        System.setProperty("gollek.gguf.fast_run.stale_daemon_force_kill", "false");
        try {
            assertFalse(GgufFastRun.staleDaemonForceKill());
        } finally {
            restoreProperty("gollek.gguf.fast_run.stale_daemon_force_kill", previous);
        }
    }

    @Test
    void staleDaemonKillWaitHasSafetyFloor() {
        String previous = System.getProperty("gollek.gguf.fast_run.stale_daemon_kill_wait_ms");
        System.setProperty("gollek.gguf.fast_run.stale_daemon_kill_wait_ms", "1");
        try {
            assertEquals(100L, GgufFastRun.staleDaemonKillWaitMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.stale_daemon_kill_wait_ms", previous);
        }
    }

    @Test
    void daemonLauncherUsesPlatformDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.clearProperty("gollek.gguf.fast_run.daemon_launcher");
        try {
            assertEquals(GgufFastRun.defaultDaemonLauncherMode(), GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherSupportsAutoOverride() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.setProperty("gollek.gguf.fast_run.daemon_launcher", " auto ");
        try {
            assertEquals("auto", GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherSupportsLaunchctlOverride() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.setProperty("gollek.gguf.fast_run.daemon_launcher", " launchctl ");
        try {
            assertEquals("launchctl", GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherFallsBackToNohupWhenInvalid() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_launcher");
        System.setProperty("gollek.gguf.fast_run.daemon_launcher", "bogus");
        try {
            assertEquals(GgufFastRun.defaultDaemonLauncherMode(), GgufFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonRequestRetryDefaultsToShortBoundedWindow() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_request_retry_ms");
        System.clearProperty("gollek.gguf.fast_run.daemon_request_retry_ms");
        try {
            assertEquals(2_000L, GgufFastRun.daemonRequestRetryMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_request_retry_ms", previous);
        }
    }

    @Test
    void daemonRequestRetryCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_request_retry_ms");
        System.setProperty("gollek.gguf.fast_run.daemon_request_retry_ms", "-1");
        try {
            assertEquals(0L, GgufFastRun.daemonRequestRetryMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_request_retry_ms", previous);
        }
    }

    @Test
    void daemonRequestRetrySleepHasSafetyFloor() {
        String previous = System.getProperty("gollek.gguf.fast_run.daemon_request_retry_sleep_ms");
        System.setProperty("gollek.gguf.fast_run.daemon_request_retry_sleep_ms", "1");
        try {
            assertEquals(10L, GgufFastRun.daemonRequestRetrySleepMillis());
        } finally {
            restoreProperty("gollek.gguf.fast_run.daemon_request_retry_sleep_ms", previous);
        }
    }

    @Test
    void prewarmArgsAddFastPathDefaults() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "b71c9d",
                        "--prompt", "where is jakarta",
                        "--max-tokens", "10",
                        "--provider", "gguf",
                        "--engine", "auto"
                },
                GgufFastRun.prewarmRunArgs(new String[] {"--model", "b71c9d"}));
    }

    @Test
    void prewarmArgsPreserveExplicitPromptAndContextTokens() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "b71c9d",
                        "--prompt", "what is jakarta",
                        "--max-tokens", "64",
                        "--provider", "gguf",
                        "--engine", "auto"
                },
                GgufFastRun.prewarmRunArgs(new String[] {
                        "--model", "b71c9d",
                        "--prompt", "what is jakarta",
                        "--max-tokens", "64"
                }));
    }

    @Test
    void prewarmArgsStripPublicCommandName() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "b71c9d",
                        "--prompt", "where is jakarta",
                        "--max-tokens", "10",
                        "--provider", "gguf",
                        "--engine", "auto"
                },
                GgufFastRun.prewarmRunArgs(new String[] {"prewarm", "--model", "b71c9d"}));
    }

    @Test
    void prewarmReturnsFallbackForLiteRtProvider() {
        assertEquals(42, GgufFastRun.prewarm(new String[] {
                "--provider", "litert",
                "--model", "7c51c9"
        }));
    }

    @Test
    void prewarmReturnsFallbackForLiteRtModelFileWithoutProvider() {
        assertEquals(42, GgufFastRun.prewarm(new String[] {
                "--model-file", "/tmp/gemma-4-E2B-it.litertlm"
        }));
    }

    @Test
    void runReturnsFallbackForLiteRtModelFileWithoutProvider() {
        assertEquals(42, GgufFastRun.run(new String[] {
                "run",
                "--model-file", "/tmp/gemma-4-E2B-it.litertlm",
                "--prompt", "where is jakarta"
        }));
    }

    @Test
    void prewarmTokenCountDefaultsToOne() {
        String previous = System.getProperty("gollek.gguf.fast_run.prewarm_tokens");
        System.clearProperty("gollek.gguf.fast_run.prewarm_tokens");
        try {
            assertEquals(1, GgufFastRun.prewarmTokenCount());
        } finally {
            restoreProperty("gollek.gguf.fast_run.prewarm_tokens", previous);
        }
    }

    @Test
    void daemonKeyMatchingIsRelaxedByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.strict_daemon_key");
        System.clearProperty("gollek.gguf.fast_run.strict_daemon_key");
        try {
            assertFalse(GgufFastRun.strictDaemonKey());
        } finally {
            restoreProperty("gollek.gguf.fast_run.strict_daemon_key", previous);
        }
    }

    @Test
    void strictDaemonKeyCanBeEnabledForDiagnostics() {
        String previous = System.getProperty("gollek.gguf.fast_run.strict_daemon_key");
        System.setProperty("gollek.gguf.fast_run.strict_daemon_key", "true");
        try {
            assertTrue(GgufFastRun.strictDaemonKey());
        } finally {
            restoreProperty("gollek.gguf.fast_run.strict_daemon_key", previous);
        }
    }

    @Test
    void hardExitAfterRunIsEnabledByDefault() {
        String previous = System.getProperty("gollek.gguf.fast_run.hard_exit_after_run");
        System.clearProperty("gollek.gguf.fast_run.hard_exit_after_run");
        try {
            assertTrue(GgufFastRun.hardExitAfterRun());
        } finally {
            restoreProperty("gollek.gguf.fast_run.hard_exit_after_run", previous);
        }
    }

    @Test
    void runCommandRoutesExplicitGgufProviderToStandaloneFastPath() {
        RunCommand command = new RunCommand();
        command.modelId = "b71c9d";
        command.prompt = "where is jakarta";
        command.providerId = "gguf";
        command.ggufEngine = "java";
        command.ggufBackend = "cpu";
        command.maxTokens = 10;
        command.temperature = 0.0;
        command.topK = 1;
        command.topP = 1.0;

        assertTrue(command.shouldTryStandaloneGgufFastPath());
        assertArrayEquals(new String[] {
                "run",
                "--model", "b71c9d",
                "--prompt", "where is jakarta",
                "--max-tokens", "10",
                "--temperature", "0.0",
                "--top-k", "1",
                "--top-p", "1.0",
                "--engine", "java",
                "--backend", "cpu",
                "--provider", "gguf"
        }, command.buildStandaloneGgufFastRunArgs());
    }

    @Test
    void runCommandDoesNotStealGgufFileFromExplicitNonGgufProvider() {
        RunCommand command = new RunCommand();
        command.modelFile = "/tmp/model.gguf";
        command.prompt = "where is jakarta";
        command.providerId = "litert";

        assertFalse(command.shouldTryStandaloneGgufFastPath());
    }

    @Test
    void runCommandRoutesGgufFileWithoutProviderToStandaloneFastPath() {
        RunCommand command = new RunCommand();
        command.modelFile = "/tmp/model.gguf";
        command.prompt = "where is jakarta";
        command.maxTokens = 1;
        command.temperature = 0.0;
        command.topK = 1;
        command.topP = 1.0;

        assertTrue(command.shouldTryStandaloneGgufFastPath());
        assertArrayEquals(new String[] {
                "run",
                "--modelFile", "/tmp/model.gguf",
                "--prompt", "where is jakarta",
                "--max-tokens", "1",
                "--temperature", "0.0",
                "--top-k", "1",
                "--top-p", "1.0",
                "--provider", "gguf"
        }, command.buildStandaloneGgufFastRunArgs());
    }

    @Test
    void runCommandJavaNativeFlagSelectsJavaGgufEngine() {
        RunCommand command = new RunCommand();
        command.modelId = "b71c9d";
        command.prompt = "where is jakarta";
        command.javaNativeGguf = true;
        command.maxTokens = 1;
        command.temperature = 0.0;
        command.topK = 1;
        command.topP = 1.0;

        assertTrue(command.shouldTryStandaloneGgufFastPath());
        assertArrayEquals(new String[] {
                "run",
                "--model", "b71c9d",
                "--prompt", "where is jakarta",
                "--max-tokens", "1",
                "--temperature", "0.0",
                "--top-k", "1",
                "--top-p", "1.0",
                "--engine", "java",
                "--provider", "gguf"
        }, command.buildStandaloneGgufFastRunArgs());
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }
}
