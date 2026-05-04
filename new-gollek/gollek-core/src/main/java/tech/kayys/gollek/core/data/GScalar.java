package tech.kayys.gollek.core.data;

public final class GScalar implements GData {
    private final String id;
    private final float value;

    public GScalar(String id, float value) {
        this.id = id;
        this.value = value;
    }

    public float value() {
        return value;
    }

    @Override
    public String id() {
        return id;
    }
}
