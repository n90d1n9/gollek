package tech.kayys.gollek.langchain4j;

import dev.langchain4j.model.chat.ChatLanguageModel;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import tech.kayys.gollek.sdk.GollekClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;

public class GollekChatModelTest {

    @Test
    void should_generate_response() {
        // Given
        GollekClient mockClient = Mockito.mock(GollekClient.class);
        GollekClient.GenerationResult mockResult = new GollekClient.GenerationResult(
                "Hello from Gollek!", 4, 1, 100);
        
        when(mockClient.generate(anyString())).thenReturn(mockResult);

        ChatLanguageModel model = GollekChatModel.builder()
                .client(mockClient)
                .build();

        // When
        String response = model.generate("Hi");

        // Then
        assertThat(response).isEqualTo("Hello from Gollek!");
    }
}
