package tech.kayys.gollek.cli.commands;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

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

    private static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
    }
}
