// ModelConfigLoader.java
package tech.kayys.gollek.spi.model.loader;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.logging.Logger;

import tech.kayys.gollek.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads ModelConfig from JSON files.
 */
public class ModelConfigLoader {
    
    private static final Logger log = Logger.getLogger(ModelConfigLoader.class.getName());
    
    private final ObjectMapper objectMapper;
    private final JsonConfigMerger jsonMerger;
    
    public ModelConfigLoader(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        this.jsonMerger = new JsonConfigMerger(objectMapper);
    }
    
    /**
     * Parse {@code config.json} from an explicit file path.
     */
    public ModelConfig load(Path configPath) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("config.json not found: " + configPath);
        }
        
        ModelConfig cfg = objectMapper.readValue(configPath.toFile(), ModelConfig.class);
        
        // Apply fallback inference for missing fields
        inferMissingFields(cfg, configPath);
        
        // Merge text_config fields
        jsonMerger.mergeTextConfig(cfg, configPath);
        
        // Apply post-load reconciliation
        postProcess(cfg);

        
        return cfg;
    }
    
    /**
     * Load {@code config.json} from the model directory.
     */
    public ModelConfig loadFromDirectory(Path modelDir) throws IOException {
        return load(modelDir.resolve("config.json"));
    }
    
    private void inferMissingFields(ModelConfig cfg, Path configPath) throws IOException {
        if ((cfg.getModelType() == null || cfg.getModelType().isBlank())
                && (cfg.getArchitectures() == null || cfg.getArchitectures().isEmpty())) {
            
            JsonNode root = objectMapper.readTree(configPath.toFile());
            String inferredType = extractTextValue(root, "model_type");
            List<String> inferredArch = extractListValue(root, "architectures");
            
            if (inferredType == null || inferredType.isBlank()) {
                inferredType = extractTextValue(root.path("text_config"), "model_type");
            }
            if (inferredArch == null || inferredArch.isEmpty()) {
                inferredArch = extractListValue(root.path("text_config"), "architectures");
            }
            
            if ((inferredType != null && !inferredType.isBlank())
                    || (inferredArch != null && !inferredArch.isEmpty())) {
                cfg.setModelType(inferredType);
                cfg.setArchitectures(inferredArch);

            }
        }
    }
    
    private void postProcess(ModelConfig cfg) {
        reconcileGemma3RopeDefaults(cfg);
        reconcileGemma4Defaults(cfg);
    }
    
    private void reconcileGemma3RopeDefaults(ModelConfig cfg) {
        if (cfg.getModelType() == null) return;
        if (!cfg.getModelType().toLowerCase().startsWith("gemma3")) return;
        
        if (cfg.getRopeThetaFull() == null) {
            cfg.setRopeThetaFull(cfg.getRopeTheta());
        }
        if (cfg.getRopeThetaSliding() == null) {
            Double localBase = cfg.getRopeLocalBaseFreq();
            cfg.setRopeThetaSliding(localBase != null ? localBase : 10000.0);
        }
    }
    
    private void reconcileGemma4Defaults(ModelConfig cfg) {
        if (cfg.getModelType() == null) return;
        if (!cfg.getModelType().toLowerCase().startsWith("gemma4")) return;
        
        // Gemma4 specific defaults
        if (cfg.getQueryPreAttnScalar() == null) {
            cfg.setQueryPreAttnScalar((double) cfg.getResolvedHeadDim());
        }
    }
    
    private String extractTextValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode()) return null;
        JsonNode v = node.get(field);
        return (v != null && v.isTextual()) ? v.asText() : null;
    }
    
    private List<String> extractListValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode()) return null;
        JsonNode v = node.get(field);
        if (v == null || !v.isArray()) return null;
        
        List<String> out = new ArrayList<>();
        v.forEach(n -> {
            if (n != null && !n.isNull()) out.add(n.asText());
        });
        return out;
    }
}
