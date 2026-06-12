package tech.kayys.gollek.runner;

import tech.kayys.gollek.spi.model.RunnerMetadata;

/**
 * Candidate runner for selection with score
 */
public record RunnerCandidate(
                String name,
                int score,
                RunnerMetadata metadata) {
}
