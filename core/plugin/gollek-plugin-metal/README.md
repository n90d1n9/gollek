# Gollek Plugin :: Metal Computation Backend

`gollek-plugin-metal` is a hardware-accelerated computation backend plugin for the Gollek ML framework designed specifically for Apple Silicon (M1/M2/M3/M4).

It implements the `ComputeBackend` Service Provider Interface (SPI) defined in `gollek-spi-tensor`.

## Architecture Flow

When `gollek-sdk-autograd` executes a tensor operation (like `Functions.Matmul.apply(a, b)`), it dynamically discovers available compute backends.
If `gollek-plugin-metal` is listed in your project dependencies, its service loader file (`META-INF/services/tech.kayys.gollek.spi.tensor.ComputeBackend`) automatically registers the `MetalComputeBackend`.

Because GPU modules override the `.priority()` method to be greater than 0, `ComputeBackendRegistry` will favor it over the standard `CpuBackend`.

## Development Roadmap

The plugin maps operations like Matrix Multiplication, Softmax, Add/Sub to `MLX` or raw `Metal Performance Shaders` via Java FFI (Foreign Function & Memory API - JEP 454). Memory pools will use off-heap segments attached directly to the Apple Unified Memory subsystem.
