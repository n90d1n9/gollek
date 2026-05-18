package tech.kayys.gollek.cli.commands;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LiteRtLmFastRunDaemonTest {

    @Test
    void staleDaemonForceKillIsEnabledByDefault() {
        String previous = System.getProperty("gollek.litert.fast_run.stale_daemon_force_kill");
        System.clearProperty("gollek.litert.fast_run.stale_daemon_force_kill");
        try {
            assertTrue(LiteRtLmFastRun.staleDaemonForceKill());
        } finally {
            restoreProperty("gollek.litert.fast_run.stale_daemon_force_kill", previous);
        }
    }

    @Test
    void staleDaemonForceKillCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.litert.fast_run.stale_daemon_force_kill");
        System.setProperty("gollek.litert.fast_run.stale_daemon_force_kill", "false");
        try {
            assertFalse(LiteRtLmFastRun.staleDaemonForceKill());
        } finally {
            restoreProperty("gollek.litert.fast_run.stale_daemon_force_kill", previous);
        }
    }

    @Test
    void staleDaemonKillWaitHasSafetyFloor() {
        String previous = System.getProperty("gollek.litert.fast_run.stale_daemon_kill_wait_ms");
        System.setProperty("gollek.litert.fast_run.stale_daemon_kill_wait_ms", "1");
        try {
            assertEquals(100L, LiteRtLmFastRun.staleDaemonKillWaitMillis());
        } finally {
            restoreProperty("gollek.litert.fast_run.stale_daemon_kill_wait_ms", previous);
        }
    }

    @Test
    void daemonProcessAliveRejectsInvalidAndCurrentPid() {
        assertFalse(LiteRtLmFastRun.isDaemonProcessAlive(-1));
        assertFalse(LiteRtLmFastRun.isDaemonProcessAlive(ProcessHandle.current().pid()));
    }

    @Test
    void strictDaemonKeyIsEnabledByDefaultForLiteRtSafety() {
        String previous = System.getProperty("gollek.litert.fast_run.strict_daemon_key");
        System.clearProperty("gollek.litert.fast_run.strict_daemon_key");
        try {
            assertTrue(LiteRtLmFastRun.strictDaemonKey());
        } finally {
            restoreProperty("gollek.litert.fast_run.strict_daemon_key", previous);
        }
    }

    @Test
    void strictDaemonKeyCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.litert.fast_run.strict_daemon_key");
        System.setProperty("gollek.litert.fast_run.strict_daemon_key", "false");
        try {
            assertFalse(LiteRtLmFastRun.strictDaemonKey());
        } finally {
            restoreProperty("gollek.litert.fast_run.strict_daemon_key", previous);
        }
    }

    @Test
    void keepAliveConversationIsDisabledByDefaultBecauseLiteRtLmAllowsOneSession() {
        String previous = System.getProperty("gollek.litert.fast_run.keepalive_conversation");
        System.clearProperty("gollek.litert.fast_run.keepalive_conversation");
        try {
            assertFalse(LiteRtLmFastRun.keepAliveConversationEnabled());
        } finally {
            restoreProperty("gollek.litert.fast_run.keepalive_conversation", previous);
        }
    }

    @Test
    void keepAliveConversationCanBeEnabledForDiagnostics() {
        String previous = System.getProperty("gollek.litert.fast_run.keepalive_conversation");
        System.setProperty("gollek.litert.fast_run.keepalive_conversation", "true");
        try {
            assertTrue(LiteRtLmFastRun.keepAliveConversationEnabled());
        } finally {
            restoreProperty("gollek.litert.fast_run.keepalive_conversation", previous);
        }
    }

    @Test
    void daemonLauncherDefaultsToLaunchctlOnMacForLiteRtPersistence() {
        String previous = System.getProperty("gollek.litert.fast_run.daemon_launcher");
        System.clearProperty("gollek.litert.fast_run.daemon_launcher");
        try {
            String expected = System.getProperty("os.name", "").toLowerCase().contains("mac")
                    ? "launchctl"
                    : "nohup";
            assertEquals(expected, LiteRtLmFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.litert.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherSupportsAutoOverride() {
        String previous = System.getProperty("gollek.litert.fast_run.daemon_launcher");
        System.setProperty("gollek.litert.fast_run.daemon_launcher", " auto ");
        try {
            assertEquals("auto", LiteRtLmFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.litert.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherSupportsLaunchctlOverride() {
        String previous = System.getProperty("gollek.litert.fast_run.daemon_launcher");
        System.setProperty("gollek.litert.fast_run.daemon_launcher", " launchctl ");
        try {
            assertEquals("launchctl", LiteRtLmFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.litert.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonLauncherFallsBackToNohupWhenInvalid() {
        String previous = System.getProperty("gollek.litert.fast_run.daemon_launcher");
        System.setProperty("gollek.litert.fast_run.daemon_launcher", "bogus");
        try {
            assertEquals("nohup", LiteRtLmFastRun.daemonLauncherMode());
        } finally {
            restoreProperty("gollek.litert.fast_run.daemon_launcher", previous);
        }
    }

    @Test
    void daemonRequestRetryDefaultsToShortBoundedWindow() {
        String previous = System.getProperty("gollek.litert.fast_run.daemon_request_retry_ms");
        System.clearProperty("gollek.litert.fast_run.daemon_request_retry_ms");
        try {
            assertEquals(2_000L, LiteRtLmFastRun.daemonRequestRetryMillis());
        } finally {
            restoreProperty("gollek.litert.fast_run.daemon_request_retry_ms", previous);
        }
    }

    @Test
    void daemonRequestRetryCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.litert.fast_run.daemon_request_retry_ms");
        System.setProperty("gollek.litert.fast_run.daemon_request_retry_ms", "-1");
        try {
            assertEquals(0L, LiteRtLmFastRun.daemonRequestRetryMillis());
        } finally {
            restoreProperty("gollek.litert.fast_run.daemon_request_retry_ms", previous);
        }
    }

    @Test
    void daemonRequestRetrySleepHasSafetyFloor() {
        String previous = System.getProperty("gollek.litert.fast_run.daemon_request_retry_sleep_ms");
        System.setProperty("gollek.litert.fast_run.daemon_request_retry_sleep_ms", "1");
        try {
            assertEquals(10L, LiteRtLmFastRun.daemonRequestRetrySleepMillis());
        } finally {
            restoreProperty("gollek.litert.fast_run.daemon_request_retry_sleep_ms", previous);
        }
    }

    @Test
    void prewarmArgsAddLiteRtDefaults() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "7c51c9",
                        "--prompt", "where is jakarta",
                        "--max-tokens", "10",
                        "--provider", "litert"
                },
                LiteRtLmFastRun.prewarmRunArgs(new String[] {"--model", "7c51c9"}));
    }

    @Test
    void prewarmArgsPreserveExplicitPromptAndBudget() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "7c51c9",
                        "--prompt", "what is jakarta",
                        "--max-tokens", "64",
                        "--provider", "litert"
                },
                LiteRtLmFastRun.prewarmRunArgs(new String[] {
                        "--model", "7c51c9",
                        "--prompt", "what is jakarta",
                        "--max-tokens", "64",
                        "--provider", "litert"
                }));
    }

    @Test
    void prewarmArgsStripPublicCommandName() {
        assertArrayEquals(
                new String[] {
                        "run",
                        "--model", "7c51c9",
                        "--prompt", "where is jakarta",
                        "--max-tokens", "10",
                        "--provider", "litert"
                },
                LiteRtLmFastRun.prewarmRunArgs(new String[] {"prewarm", "--model", "7c51c9"}));
    }

    @Test
    void prewarmTokenCountDefaultsToEngineOpenOnly() {
        String previous = System.getProperty("gollek.litert.fast_run.prewarm_tokens");
        System.clearProperty("gollek.litert.fast_run.prewarm_tokens");
        try {
            assertEquals(0, LiteRtLmFastRun.prewarmTokenCount());
        } finally {
            restoreProperty("gollek.litert.fast_run.prewarm_tokens", previous);
        }
    }

    @Test
    void prewarmTokenCountAllowsDecodeWarmupOverride() {
        String previous = System.getProperty("gollek.litert.fast_run.prewarm_tokens");
        System.setProperty("gollek.litert.fast_run.prewarm_tokens", "2");
        try {
            assertEquals(2, LiteRtLmFastRun.prewarmTokenCount());
        } finally {
            restoreProperty("gollek.litert.fast_run.prewarm_tokens", previous);
        }
    }

    @Test
    void prewarmTokenCountHasZeroFloor() {
        String previous = System.getProperty("gollek.litert.fast_run.prewarm_tokens");
        System.setProperty("gollek.litert.fast_run.prewarm_tokens", "-1");
        try {
            assertEquals(0, LiteRtLmFastRun.prewarmTokenCount());
        } finally {
            restoreProperty("gollek.litert.fast_run.prewarm_tokens", previous);
        }
    }

    @Test
    void prewarmIterationCountDefaultsToThree() {
        String previous = System.getProperty("gollek.litert.fast_run.prewarm_iterations");
        System.clearProperty("gollek.litert.fast_run.prewarm_iterations");
        try {
            assertEquals(3, LiteRtLmFastRun.prewarmIterationCount());
        } finally {
            restoreProperty("gollek.litert.fast_run.prewarm_iterations", previous);
        }
    }

    @Test
    void prewarmIterationCountHasSafetyFloor() {
        String previous = System.getProperty("gollek.litert.fast_run.prewarm_iterations");
        System.setProperty("gollek.litert.fast_run.prewarm_iterations", "0");
        try {
            assertEquals(1, LiteRtLmFastRun.prewarmIterationCount());
        } finally {
            restoreProperty("gollek.litert.fast_run.prewarm_iterations", previous);
        }
    }

    @Test
    void dynamicEngineTokensAreEnabledByDefaultForShortRuns() {
        String previous = System.getProperty("gollek.litert.fast_run.dynamic_engine_tokens");
        System.clearProperty("gollek.litert.fast_run.dynamic_engine_tokens");
        try {
            assertTrue(LiteRtLmFastRun.dynamicEngineTokensEnabled());
        } finally {
            restoreProperty("gollek.litert.fast_run.dynamic_engine_tokens", previous);
        }
    }

    @Test
    void dynamicEngineTokensCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.litert.fast_run.dynamic_engine_tokens");
        System.setProperty("gollek.litert.fast_run.dynamic_engine_tokens", "false");
        try {
            assertFalse(LiteRtLmFastRun.dynamicEngineTokensEnabled());
        } finally {
            restoreProperty("gollek.litert.fast_run.dynamic_engine_tokens", previous);
        }
    }

    @Test
    void shortDynamicEngineTokensDefaultToMeasuredStableFloor() {
        String previousMin = System.getProperty("gollek.litert.fast_run.min_engine_tokens");
        String previousMax = System.getProperty("gollek.litert.fast_run.max_engine_tokens");
        System.clearProperty("gollek.litert.fast_run.min_engine_tokens");
        System.clearProperty("gollek.litert.fast_run.max_engine_tokens");
        try {
            assertEquals(256, LiteRtLmFastRun.dynamicEngineTokenBudget("where is jakarta", 10));
        } finally {
            restoreProperty("gollek.litert.fast_run.min_engine_tokens", previousMin);
            restoreProperty("gollek.litert.fast_run.max_engine_tokens", previousMax);
        }
    }

    @Test
    void bareQuestionPromptsAreNormalizedForGemmaChatQuality() {
        assertEquals("Where is jakarta?", LiteRtLmFastRun.promptForModel("where is jakarta"));
    }

    @Test
    void promptNormalizationLeavesPunctuatedAndNonQuestionPromptsAlone() {
        assertEquals("Where is Jakarta?", LiteRtLmFastRun.promptForModel("Where is Jakarta?"));
        assertEquals("write a poem", LiteRtLmFastRun.promptForModel("write a poem"));
    }

    @Test
    void promptNormalizationCanBeDisabledForDiagnostics() {
        String previous = System.getProperty("gollek.litert.fast_run.normalize_short_questions");
        System.setProperty("gollek.litert.fast_run.normalize_short_questions", "false");
        try {
            assertEquals("where is jakarta", LiteRtLmFastRun.promptForModel("where is jakarta"));
        } finally {
            restoreProperty("gollek.litert.fast_run.normalize_short_questions", previous);
        }
    }

    @Test
    void warmupStateTracksOnlyMissingIterations() {
        LiteRtLmFastRun.WarmupState state = new LiteRtLmFastRun.WarmupState();

        assertEquals(3, state.remainingIterations(3));
        state.markCompleted(1);
        assertEquals(2, state.remainingIterations(3));
        state.markCompleted(3);
        assertEquals(0, state.remainingIterations(3));
    }

    @Test
    void warmupStateNeverRegresses() {
        LiteRtLmFastRun.WarmupState state = new LiteRtLmFastRun.WarmupState();

        state.markCompleted(3);
        state.markCompleted(1);

        assertEquals(0, state.remainingIterations(3));
        assertEquals(2, state.remainingIterations(5));
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }
}
