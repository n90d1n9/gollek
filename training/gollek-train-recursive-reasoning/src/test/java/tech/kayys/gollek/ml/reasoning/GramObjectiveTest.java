package tech.kayys.gollek.ml.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GramObjectiveTest {
    @Test
    void aggregatesWeightedSurrogateElboTermsWithSumReduction() {
        GramObjectiveConfig config = GramObjectiveConfig.builder()
                .klBeta(0.1)
                .klBalance(0.8)
                .latentProcessRewardWeight(0.5)
                .adaptiveComputationWeight(0.25)
                .reduction(GramObjectiveReduction.SUM)
                .build();

        GramObjectiveBreakdown breakdown = GramObjective.evaluate(
                List.of(
                        new GramSupervisionStepLoss(0, 2.0, 10.0, 1.0, 4.0, Map.of("step", "a")),
                        new GramSupervisionStepLoss(1, 4.0, 2.0, 3.0, 0.0, Map.of("step", "b"))),
                config);

        assertEquals(6.0, breakdown.reconstructionNll(), 1e-9);
        assertEquals(12.0, breakdown.klDivergence(), 1e-9);
        assertEquals(4.0, breakdown.latentProcessRewardLoss(), 1e-9);
        assertEquals(4.0, breakdown.adaptiveComputationLoss(), 1e-9);
        assertEquals(10.2, breakdown.totalLoss(), 1e-9);
        assertEquals("gram-surrogate-elbo", breakdown.metadata().get("objective"));
        assertTrue((Boolean) breakdown.metadata().get("truncatedSurrogate"));
    }

    @Test
    void meanReductionAveragesTermsAcrossDeepSupervisionSteps() {
        GramObjectiveConfig config = GramObjectiveConfig.builder()
                .klBeta(0.2)
                .klBalance(0.5)
                .reduction(GramObjectiveReduction.MEAN)
                .build();

        GramObjectiveBreakdown breakdown = GramObjective.evaluate(
                List.of(
                        new GramSupervisionStepLoss(0, 2.0, 10.0, 0.0, 0.0, Map.of()),
                        new GramSupervisionStepLoss(1, 4.0, 2.0, 0.0, 0.0, Map.of())),
                config);

        assertEquals(3.0, breakdown.reconstructionNll(), 1e-9);
        assertEquals(6.0, breakdown.klDivergence(), 1e-9);
        assertEquals(4.2, breakdown.totalLoss(), 1e-9);
    }

    @Test
    void rejectsInvalidWeightsAndLossTerms() {
        assertThrows(IllegalArgumentException.class, () -> GramObjectiveConfig.builder().klBalance(1.5).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> new GramSupervisionStepLoss(0, Double.NaN, 0.0, 0.0, 0.0, Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> GramObjective.evaluate(List.of(), GramObjectiveConfig.builder().build()));
    }
}
