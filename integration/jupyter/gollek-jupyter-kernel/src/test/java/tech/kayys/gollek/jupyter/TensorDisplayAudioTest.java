package tech.kayys.gollek.jupyter;

import org.dflib.jjava.jupyter.kernel.display.DisplayData;
import org.dflib.jjava.jupyter.kernel.display.mime.MIMEType;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorDisplayAudioTest {

    @Test
    void renderSupportsNotebookAudioClip() throws Exception {
        NotebookAudioClip clip = NotebookAudioClip.ofWav("Demo Tone", 8000, 1, wavBytes());

        DisplayData data = TensorDisplay.render(clip, null);

        assertNotNull(data);
        assertTrue(data.getData(MIMEType.TEXT_PLAIN).toString().contains("Audio(title=Demo Tone, sampleRate=8000, channels=1"));
        assertTrue(data.hasDataForType(MIMEType.TEXT_HTML));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("<audio controls"));
        assertTrue(data.getData(MIMEType.TEXT_HTML).toString().contains("data:audio/wav;base64,"));
        assertTrue(data.getData(MIMEType.parse("audio/wav")).toString().length() > 10);
    }

    private static byte[] wavBytes() {
        byte[] pcm = new byte[16];
        ByteBuffer buffer = ByteBuffer.allocate(44 + pcm.length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(new byte[]{'R', 'I', 'F', 'F'});
        buffer.putInt(36 + pcm.length);
        buffer.put(new byte[]{'W', 'A', 'V', 'E'});
        buffer.put(new byte[]{'f', 'm', 't', ' '});
        buffer.putInt(16);
        buffer.putShort((short) 1);
        buffer.putShort((short) 1);
        buffer.putInt(8000);
        buffer.putInt(16000);
        buffer.putShort((short) 2);
        buffer.putShort((short) 16);
        buffer.put(new byte[]{'d', 'a', 't', 'a'});
        buffer.putInt(pcm.length);
        buffer.put(pcm);
        return buffer.array();
    }
}
