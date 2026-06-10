package tech.kayys.gollek.ml.bytelatent;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Shared compact formatter for ranked byte-latent next-token previews.
 */
public final class ByteLatentPreviewFormatter {

    private ByteLatentPreviewFormatter() {
    }

    public static List<String> nextTokensPreview(List<ByteLatentTokenCandidate> nextTokens, int limit) {
        if (nextTokens == null || nextTokens.isEmpty() || limit <= 0) {
            return List.of();
        }
        int size = Math.min(limit, nextTokens.size());
        List<String> preview = new ArrayList<>(size);
        for (int index = 0; index < size; index++) {
            ByteLatentTokenCandidate candidate = nextTokens.get(index);
            preview.add(candidate.rank() + ":" + candidate.tokenId() + "@"
                    + String.format(Locale.ROOT, "%.4f", candidate.score()));
        }
        return List.copyOf(preview);
    }
}
