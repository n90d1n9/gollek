package tech.kayys.gollek.core.tensor;

public interface Tensor {
    Shape shape();
    Device device();
    DType dtype();
    
    Tensor add(Tensor other);
    Tensor mul(Tensor other);
    Tensor mul(float scalar);
    Tensor matmul(Tensor other);
    Tensor reshape(long... newShape);
    
    default void release() {}
    
    // Minimal factory to allow compilation
    static Tensor zeros(long... shape) {
        return null; 
    }
}
