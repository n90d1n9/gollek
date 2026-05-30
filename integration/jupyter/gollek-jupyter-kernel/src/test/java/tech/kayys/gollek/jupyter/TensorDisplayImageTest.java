package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorDisplayImageTest {

    @Test
    void renderSupportsBufferedImagePreview() {
        BufferedImage image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, Color.ORANGE.getRGB());
        image.setRGB(1, 0, Color.BLUE.getRGB());
        image.setRGB(0, 1, Color.BLUE.getRGB());
        image.setRGB(1, 1, Color.ORANGE.getRGB());

        DisplayData data = TensorDisplay.render(image, null);

        assertNotNull(data);
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("Image(width=4, height=4"));
        assertTrue(data.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("data:image/png;base64,"));
        assertTrue(data.hasDataForType(MIMEType.IMAGE_PNG));
    }
}
