package tech.kayys.gollek.ml.bytelatent;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class ByteLatentPreviewFormatterTest {

    @Test
    void formatsCompactRankedPreview() {
        List<String> preview = ByteLatentPreviewFormatter.nextTokensPreview(
                List.of(
                        new ByteLatentTokenCandidate(47, 1.0d, 1),
                        new ByteLatentTokenCandidate(48, 0.5d, 2),
                        new ByteLatentTokenCandidate(49, 1.0d / 3.0d, 3)),
                3);

        assertEquals(List.of(
                "1:47@1.0000",
                "2:48@0.5000",
                "3:49@0.3333"), preview);
    }

    @Test
    void trimsAndHandlesEmptyInput() {
        List<String> trimmed = ByteLatentPreviewFormatter.nextTokensPreview(
                List.of(
                        new ByteLatentTokenCandidate(10, 0.9d, 1),
                        new ByteLatentTokenCandidate(11, 0.4d, 2)),
                1);

        assertEquals(List.of("1:10@0.9000"), trimmed);
        assertEquals(List.of(), ByteLatentPreviewFormatter.nextTokensPreview(List.of(), 3));
        assertEquals(List.of(), ByteLatentPreviewFormatter.nextTokensPreview(null, 3));
        assertEquals(List.of(), ByteLatentPreviewFormatter.nextTokensPreview(trimmedCandidates(), 0));
    }

    private static List<ByteLatentTokenCandidate> trimmedCandidates() {
        return List.of(
                new ByteLatentTokenCandidate(10, 0.9d, 1),
                new ByteLatentTokenCandidate(11, 0.4d, 2));
    }
}
