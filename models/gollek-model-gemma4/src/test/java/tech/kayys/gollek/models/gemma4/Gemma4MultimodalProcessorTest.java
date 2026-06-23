package tech.kayys.gollek.models.gemma4;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;

import java.util.Base64;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class Gemma4MultimodalProcessorTest {

    @Test
    void testTextOnlyProcessing() {
        Gemma4MultimodalProcessor processor = new Gemma4MultimodalProcessor();
        
        MultimodalRequest request = new MultimodalRequest();
        request.setInputs(new MultimodalContent[]{
                MultimodalContent.ofText("Hello world")
        });

        MultimodalResponse response = processor.process(request).await().indefinitely();
        assertNotNull(response);
        assertEquals("Processed Hello world", response.getOutputs()[0].getText());
        assertTrue(response.getMetadata().isEmpty());
    }

    @Test
    void testImageProcessingWithCaching() {
        Gemma4MultimodalProcessor processor = new Gemma4MultimodalProcessor();
        
        // 1x1 pixel white JPEG encoded in base64
        String dummyBase64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=";
        
        MultimodalRequest request = new MultimodalRequest();
        request.setInputs(new MultimodalContent[]{
                MultimodalContent.ofText("What is this?"),
                MultimodalContent.ofBase64Image(Base64.getDecoder().decode(dummyBase64Image), "image/jpeg")
        });

        // First process
        MultimodalResponse response1 = processor.process(request).await().indefinitely();
        assertNotNull(response1);
        Map<String, Object> meta1 = response1.getMetadata();
        assertEquals(1, meta1.size());
        
        // Second process with same image
        MultimodalResponse response2 = processor.process(request).await().indefinitely();
        assertNotNull(response2);
        Map<String, Object> meta2 = response2.getMetadata();
        assertEquals(1, meta2.size());
        
        // The tensors in metadata should be exact same instances because of caching
        Object tensor1 = meta1.values().iterator().next();
        Object tensor2 = meta2.values().iterator().next();
        assertSame(tensor1, tensor2, "Tensors should be cached and identically referenced");
    }
}
