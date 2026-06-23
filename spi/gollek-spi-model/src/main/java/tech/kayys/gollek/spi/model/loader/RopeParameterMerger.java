// RopeParameterMerger.java
package tech.kayys.gollek.spi.model.loader;

import com.fasterxml.jackson.databind.JsonNode;
import tech.kayys.gollek.spi.model.ModelConfig;

/**
 * Merges rope_parameters from JSON into ModelConfig.
 */
public class RopeParameterMerger {
    
    public void merge(ModelConfig cfg, JsonNode ropeParams) {
        if (ropeParams == null || !ropeParams.isObject()) {
            return;
        }
        
        // Root params
        Double rootTheta = doubleValue(ropeParams, "rope_theta");
        if (rootTheta != null) {
            cfg.setRopeTheta(rootTheta);
        }
        
        Double rootPartial = doubleValue(ropeParams, "partial_rotary_factor");
        if (rootPartial != null) {
            cfg.setPartialRotaryFactor(rootPartial);
        }
        
        String rootType = textValue(ropeParams, "rope_type");
        if (rootType != null) {
            ModelConfig.RopeScaling scaling = cfg.getRopeScaling();
            if (scaling == null) {
                scaling = new ModelConfig.RopeScaling();
                cfg.setRopeScaling(scaling);
            }
            scaling.setType(rootType);
        }
        
        // Full attention params
        JsonNode full = ropeParams.get("full_attention");
        if (full != null && full.isObject()) {
            Double v = doubleValue(full, "rope_theta");
            if (v != null) {
                cfg.setRopeThetaFull(v);
            }
            Double p = doubleValue(full, "partial_rotary_factor");
            if (p != null) {
                cfg.setPartialRotaryFactorFull(p);
            }
            String rt = textValue(full, "rope_type");
            if (rt != null) {
                cfg.setRopeTypeFull(rt);
            }
        }
        
        // Sliding attention params
        JsonNode sliding = ropeParams.get("sliding_attention");
        if (sliding != null && sliding.isObject()) {
            Double v = doubleValue(sliding, "rope_theta");
            if (v != null) {
                cfg.setRopeThetaSliding(v);
            }
            Double p = doubleValue(sliding, "partial_rotary_factor");
            if (p != null) {
                cfg.setPartialRotaryFactorSliding(p);
            }
            String rt = textValue(sliding, "rope_type");
            if (rt != null) {
                cfg.setRopeTypeSliding(rt);
            }
        }
    }
    
    private Double doubleValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asDouble() : null;
    }
    
    private String textValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isTextual() ? v.asText() : v.toString();
    }
}
