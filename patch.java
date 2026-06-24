        try {
            return FlashAttentionDenseFallbackLoop.compute(
                    q, source, config, layerIdx, startPos, numQHeads, numKVHeads, headDim, scale, causal, softCap,
                    attentionContextBuffer);
        } finally {
            java.lang.ref.Reference.reachabilityFence(q);
            java.lang.ref.Reference.reachabilityFence(k);
            java.lang.ref.Reference.reachabilityFence(v);
        }
