package tech.kayys.gollek.spi.model;

/**
 * A machine-readable model-family plugin contract problem.
 */
public record ModelFamilyContractViolation(
        String familyId,
        String code,
        String message) {

    public ModelFamilyContractViolation {
        familyId = familyId == null || familyId.isBlank() ? "unknown" : familyId.trim();
        code = code == null || code.isBlank() ? "contract_violation" : code.trim();
        message = message == null || message.isBlank() ? code : message.trim();
    }

    public String summary() {
        return familyId + "[" + code + "]: " + message;
    }
}
