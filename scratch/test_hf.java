
import tech.kayys.gollek.model.repo.hf.HuggingFaceClient;
import tech.kayys.gollek.model.repo.hf.HuggingFaceConfig;
import tech.kayys.gollek.model.repo.hf.HuggingFaceModelInfo;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Optional;

public class TestHFListing {
    public static void main(String[] args) throws Exception {
        HuggingFaceConfig config = new HuggingFaceConfig() {
            public String baseUrl() { return "https://huggingface.co"; }
            public Optional<String> token() { return Optional.empty(); }
            public String userAgent() { return "Gollek/0.1.0"; }
            public String revision() { return "main"; }
            public int timeoutSeconds() { return 60; }
        };
        
        HuggingFaceClient client = new HuggingFaceClient();
        // Since it's CDI, we manually inject or use a mock if possible
        // For a quick scratch, I'll just look at the code again.
    }
}
