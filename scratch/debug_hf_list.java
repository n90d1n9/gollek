///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS tech.kayys.gollek:gollek-model-repo-hf:0.1.0-SNAPSHOT
//DEPS tech.kayys.gollek:gollek-spi:0.1.0-SNAPSHOT
//DEPS com.fasterxml.jackson.core:jackson-databind:2.15.2
//DEPS com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.15.2

import tech.kayys.gollek.model.repo.hf.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;

public class debug_hf_list {
    public static void main(String[] args) throws Exception {
        String modelId = "CompVis/stable-diffusion-v1-4";
        
        HuggingFaceConfig config = new HuggingFaceConfig() {
            @Override public String baseUrl() { return "https://huggingface.co"; }
            @Override public String apiBaseUrl() { return "https://huggingface.co"; }
            @Override public java.util.Optional<String> token() { return java.util.Optional.empty(); }
            @Override public String userAgent() { return "Gollek/0.1.0"; }
            @Override public int timeoutSeconds() { return 30; }
            @Override public String revision() { return "main"; }
            @Override public boolean autoDownload() { return true; }
        };

        HuggingFaceClient client = new HuggingFaceClient();
        client.config = config;
        client.objectMapper = new ObjectMapper();
        client.objectMapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());

        // This should now include ?siblings=true because of my fix
        List<String> files = client.listFiles(modelId);
        System.out.println("Total files found: " + files.size());
        files.stream().filter(f -> f.contains("/")).limit(20).forEach(System.out::println);
    }
}
