package tech.kayys.gollek.spi.model;

/**
 * Direct SafeTensor execution readiness for a detachable model-family plugin.
 */
public enum ModelFamilyDirectSupport {
    READY,
    EXPERIMENTAL,
    DECLARED_NO_ADAPTER,
    PENDING,
    NOT_APPLICABLE,
    NOT_ADVERTISED;

    public boolean ready() {
        return this == READY;
    }

    public String label() {
        return name().toLowerCase();
    }
}
