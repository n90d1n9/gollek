package tech.kayys.gollek.ml.autograd;

/**
 * Minimal autograd callback surface kept for ML/trainer compatibility.
 */
public abstract class Function {

    private final String name;

    protected Function(String name) {
        this.name = name;
    }

    public final String name() {
        return name;
    }

    public abstract void backward(GradTensor upstream);

    public abstract static class Context extends Function {
        protected Context(String name) {
            super(name);
        }
    }
}
