/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class ResolvedModelWeightsCandidateTest {

    @Test
    void resolvesModelAndLayerWeightsFromArchitectureCandidates() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "candidate-test",
                  "num_hidden_layers": 1,
                  "tie_word_embeddings": false
                }
                """, ModelConfig.class);
        AccelTensor embed = AccelTensor.zeros(4, 3);
        AccelTensor lmHead = AccelTensor.zeros(4, 3);
        AccelTensor finalNorm = AccelTensor.zeros(3);
        AccelTensor query = AccelTensor.zeros(3, 3);
        AccelTensor attentionNorm = AccelTensor.zeros(3);
        AccelTensor ffnNorm = AccelTensor.zeros(3);
        AccelTensor layerScalar = AccelTensor.zeros(1);
        layerScalar.setFlat(0, 0.75f);

        Map<String, AccelTensor> weights = new HashMap<>();
        weights.put("fallback.embed", embed);
        weights.put("fallback.lm_head", lmHead);
        weights.put("fallback.final_norm", finalNorm);
        weights.put("fallback.layer.0.q", query);
        weights.put("fallback.layer.0.attn_norm", attentionNorm);
        weights.put("fallback.layer.0.ffn_norm", ffnNorm);
        weights.put("fallback.layer.0.scalar", layerScalar);

        ResolvedModelWeights resolved = ResolvedModelWeights.create(
                weights, config, new CandidateArchitecture(), false);

        assertSame(embed, resolved.embedTokens());
        assertSame(lmHead, resolved.lmHead());
        assertSame(finalNorm, resolved.finalNorm());
        assertSame(query, resolved.layer(0).queryWeight());
        assertSame(attentionNorm, resolved.layer(0).attentionNormWeight());
        assertSame(ffnNorm, resolved.layer(0).preFfnNormWeight());
        assertSame(layerScalar, resolved.layer(0).layerScalarWeight());
        assertEquals(0.75f, resolved.layer(0).layerScalarValue(), 0.0001f);
    }

    @Test
    void resolvesFusedQueryKeyValueWeightsFromFusedCandidates() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "fused-candidate-test",
                  "num_hidden_layers": 1
                }
                """, ModelConfig.class);
        AccelTensor fusedQkv = AccelTensor.zeros(8, 4);
        Map<String, AccelTensor> weights = Map.of("fallback.layer.0.qkv", fusedQkv);

        ResolvedModelWeights resolved = ResolvedModelWeights.create(
                weights, config, new FusedCandidateArchitecture(), false);

        assertSame(fusedQkv, resolved.layer(0).queryWeight());
        assertSame(fusedQkv, resolved.layer(0).keyWeight());
        assertSame(fusedQkv, resolved.layer(0).valueWeight());
    }

    private static final class CandidateArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "candidate";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("CandidateForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("candidate-test");
        }

        @Override
        public String embedTokensWeight() {
            return "primary.embed";
        }

        @Override
        public List<String> embedTokensWeightCandidates() {
            return List.of("primary.embed", "fallback.embed");
        }

        @Override
        public String finalNormWeight() {
            return "primary.final_norm";
        }

        @Override
        public List<String> finalNormWeightCandidates() {
            return List.of("primary.final_norm", "fallback.final_norm");
        }

        @Override
        public String lmHeadWeight() {
            return "primary.lm_head";
        }

        @Override
        public List<String> lmHeadWeightCandidates() {
            return List.of("primary.lm_head", "fallback.lm_head");
        }

        @Override
        public String layerQueryWeight(int i) {
            return "primary.layer.%d.q".formatted(i);
        }

        @Override
        public List<String> layerQueryWeightCandidates(int i) {
            return List.of(layerQueryWeight(i), "fallback.layer.%d.q".formatted(i));
        }

        @Override
        public String layerKeyWeight(int i) {
            return "primary.layer.%d.k".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "primary.layer.%d.v".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "primary.layer.%d.o".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "primary.layer.%d.attn_norm".formatted(i);
        }

        @Override
        public List<String> layerAttentionNormWeightCandidates(int i) {
            return List.of(layerAttentionNormWeight(i), "fallback.layer.%d.attn_norm".formatted(i));
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "primary.layer.%d.ffn_gate".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "primary.layer.%d.ffn_up".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "primary.layer.%d.ffn_down".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "primary.layer.%d.ffn_norm".formatted(i);
        }

        @Override
        public List<String> layerFfnNormWeightCandidates(int i) {
            return List.of(layerFfnNormWeight(i), "fallback.layer.%d.ffn_norm".formatted(i));
        }

        @Override
        public String layerScalarWeight(int i) {
            return "primary.layer.%d.scalar".formatted(i);
        }

        @Override
        public List<String> layerScalarWeightCandidates(int i) {
            return List.of(layerScalarWeight(i), "fallback.layer.%d.scalar".formatted(i));
        }
    }

    private static final class FusedCandidateArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "fused-candidate";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("FusedCandidateForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("fused-candidate-test");
        }

        @Override
        public String embedTokensWeight() {
            return "model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "model.norm.weight";
        }

        @Override
        public boolean hasFusedQKV() {
            return true;
        }

        @Override
        public String layerFusedQKVWeight(int i) {
            return "primary.layer.%d.qkv".formatted(i);
        }

        @Override
        public List<String> layerFusedQKVWeightCandidates(int i) {
            return List.of(layerFusedQKVWeight(i), "fallback.layer.%d.qkv".formatted(i));
        }

        @Override
        public String layerQueryWeight(int i) {
            return "unused.layer.%d.q".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "unused.layer.%d.k".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "unused.layer.%d.v".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "model.layers.%d.mlp.gate_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "model.layers.%d.mlp.up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }
    }
}
