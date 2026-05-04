package tech.kayys.gollek.engine.inference;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import tech.kayys.gollek.engine.routing.ModelRouterService;
import tech.kayys.gollek.metrics.MetricsPublisher;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.InferenceStage;
import tech.kayys.gollek.spi.Message;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class StageAwareOrchestratorTest {

    @Mock
    private ModelRouterService router;

    @Mock
    private MetricsPublisher metrics;

    @Mock
    private tech.kayys.gollek.prefilldecode.DisaggregatedLLMProvider pdProvider;

    private DefaultInferenceOrchestrator orchestrator;

    @BeforeEach
    void setUp() {
        orchestrator = new DefaultInferenceOrchestrator(router, metrics);
        orchestrator.setPdProvider(pdProvider);
        // Default: disaggregated mode disabled
        orchestrator.setDisaggregatedMode(false);
    }

    @Test
    void whenDisaggregatedModeDisabled_stagesResolveToCombined() {
        // Arrange
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user("Short generic prompt"))
                .build();

        when(router.route(any(), any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<InferenceRequest> captor = ArgumentCaptor.forClass(InferenceRequest.class);
        verify(router).route(eq("llama-3-8b"), captor.capture());

        InferenceRequest routedRequest = captor.getValue();
        assertThat(routedRequest.getInferenceStage()).isEqualTo(InferenceStage.COMBINED);
    }

    @Test
    void whenDisaggregatedModeEnabled_smallPrompt_resolvesToCombined() {
        // Arrange
        orchestrator.setDisaggregatedMode(true);
        orchestrator.setSmallPromptThreshold(100);

        // "Short prompt" is ~12 chars -> ~3 tokens << 100
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user("Short prompt"))
                .build();

        when(router.route(any(), any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<InferenceRequest> captor = ArgumentCaptor.forClass(InferenceRequest.class);
        verify(router).route(eq("llama-3-8b"), captor.capture());

        assertThat(captor.getValue().getInferenceStage()).isEqualTo(InferenceStage.COMBINED);
    }

    @Test
    void whenDisaggregatedModeEnabled_largePrompt_resolvesToPrefill() {
        // Arrange
        orchestrator.setDisaggregatedMode(true);
        orchestrator.setSmallPromptThreshold(5);

        // "This is a longer prompt that should trigger prefill stage"
        // 60 chars -> ~15 tokens > 5
        String longText = "This is a longer prompt that should trigger prefill stage routing logic.";
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user(longText))
                .build();

        when(pdProvider.infer(any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<tech.kayys.gollek.spi.provider.ProviderRequest> captor = ArgumentCaptor.forClass(tech.kayys.gollek.spi.provider.ProviderRequest.class);
        verify(pdProvider).infer(captor.capture());

        tech.kayys.gollek.spi.provider.ProviderRequest routed = captor.getValue();
        verifyNoInteractions(router);
    }

    @Test
    void explicitStageIsRespected() {
        // Arrange
        orchestrator.setDisaggregatedMode(true);

        // Explicitly set DECODE stage even if it looks like a prefill
        InferenceRequest request = InferenceRequest.builder()
                .model("llama-3-8b")
                .message(Message.user("Some prompt"))
                .inferenceStage(InferenceStage.DECODE)
                .build();

        when(pdProvider.infer(any())).thenReturn(Uni.createFrom().item(mock(InferenceResponse.class)));

        // Act
        orchestrator.executeAsync("llama-3-8b", request).await().indefinitely();

        // Assert
        ArgumentCaptor<tech.kayys.gollek.spi.provider.ProviderRequest> captor = ArgumentCaptor.forClass(tech.kayys.gollek.spi.provider.ProviderRequest.class);
        verify(pdProvider).infer(captor.capture());
        verifyNoInteractions(router);
    }
}
