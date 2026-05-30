package tech.kayys.gollek.spi.model;

import java.util.List;

/**
 * Duplicate model-type claim detected across detachable model-family plugins.
 */
public record ModelFamilyClaimConflict(
        String claimType,
        String claim,
        List<String> familyIds) {

    public ModelFamilyClaimConflict {
        claimType = claimType == null || claimType.isBlank() ? "model_type" : claimType.trim();
        claim = claim == null ? "" : claim.trim();
        familyIds = familyIds == null ? List.of() : List.copyOf(familyIds);
    }

    public String summary() {
        return claimType + " " + claim + " -> " + String.join(", ", familyIds);
    }
}
