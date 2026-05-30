package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed execution playbook layered on top of a round-history recommendation.
 */
public record DiffusionOpdRoundHistoryPlaybook(
        DiffusionOpdRoundHistoryRecommendation recommendation,
        Boolean reviewRequired,
        Integer cooldownRounds,
        Integer nextCheckWindowSize,
        String escalationTarget,
        List<String> checklist) {
}
