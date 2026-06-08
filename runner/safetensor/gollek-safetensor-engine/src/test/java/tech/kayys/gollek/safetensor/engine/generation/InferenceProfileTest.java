/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InferenceProfileTest {

    @Test
    void exposesLogitsPathSeparatelyFromLinearPathEvidence() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        try {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");

            DirectInferenceProfiler.recordLinearPath("attn_q_proj", "matvec");
            DirectInferenceProfiler.recordLinearPath("logits", "metal_matmul_f16");
            DirectInferenceProfiler.recordLinearPath("logits", "metal_matmul_f16");

            String summary = profile.summary("metal");
            assertTrue(summary.contains("linear_paths={logits:metal_matmul_f16=2, attn_q_proj:matvec=1}"));
            assertTrue(summary.contains("logits_paths={metal_matmul_f16=2}"));

            Map<String, Object> metadata = profile.metadata("metal");
            assertEquals(2, metadata.get("profile_linear_path_logits_metal_matmul_f16_count"));
            assertEquals(2, metadata.get("profile_logits_path_metal_matmul_f16_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            if (previousProfile == null) {
                System.clearProperty("gollek.profile");
            } else {
                System.setProperty("gollek.profile", previousProfile);
            }
        }
    }
}
