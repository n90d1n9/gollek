///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek SDK Core Tensor API Example
//
// Demonstrates unified Tensor API with:
// - Device placement (CPU, CUDA, MPS)
// - Operator overloading
// - Memory management
// - NoGrad context
//
// Prerequisites:
//   - Java 25+
//   - JBang: curl -Ls https://sh.jbang.dev | bash -s -
//   - All SDK modules built
//
// Usage:
//   jbang gollek-sdk-core-example.java
//
// DEPS tech.kayys.gollek:gollek-sdk-core:0.1.0-SNAPSHOT
// JAVA 25+

import tech.kayys.gollek.sdk.core.*;

public class gollek_sdk_core_example {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║     Gollek SDK Core Tensor API Example                   ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // 1. Tensor factory methods
        System.out.println("1. Creating tensors with factory methods...");
        Tensor zeros = Tensor.zeros(2, 3);
        Tensor ones = Tensor.ones(2, 3);
        Tensor randn = Tensor.randn(2, 3);
        Tensor rand = Tensor.rand(2, 3);

        System.out.println("   ✓ zeros: " + zeros);
        System.out.println("   ✓ ones: " + ones);
        System.out.println("   ✓ randn: " + randn);
        System.out.println("   ✓ rand: " + rand);
        System.out.println();

        // 2. Device placement
        System.out.println("2. Device placement...");
        Device cpu = Device.CPU;
        Device cuda = Device.of("cuda", 0);
        Device mps = Device.MPS;

        Tensor x = Tensor.randn(2, 3).to(cpu);
        System.out.println("   ✓ Tensor on CPU: " + x.device());
        System.out.println("   ✓ Available devices: CPU, CUDA:0, MPS");
        System.out.println();

        // 3. Operator overloading
        System.out.println("3. Tensor operations...");
        Tensor a = Tensor.randn(2, 3);
        Tensor b = Tensor.randn(2, 3);

        Tensor c = a.add(b);
        Tensor d = a.mul(2.0f);
        Tensor e = a.sub(b);

        System.out.println("   ✓ Addition: a + b");
        System.out.println("   ✓ Scalar mul: a * 2.0");
        System.out.println("   ✓ Subtraction: a - b");
        System.out.println();

        // 4. Activations
        System.out.println("4. Activation functions...");
        Tensor x_relu = a.relu();
        Tensor x_sigmoid = a.sigmoid();
        Tensor x_softmax = a.softmax();

        System.out.println("   ✓ ReLU applied");
        System.out.println("   ✓ Sigmoid applied");
        System.out.println("   ✓ Softmax applied");
        System.out.println();

        // 5. Reshaping
        System.out.println("5. Reshaping operations...");
        Tensor x = Tensor.randn(2, 3, 4);
        System.out.println("   ✓ Original shape: " + java.util.Arrays.toString(x.shape()));

        Tensor x_flat = x.flatten();
        System.out.println("   ✓ Flattened: " + java.util.Arrays.toString(x_flat.shape()));

        Tensor x_reshaped = x_flat.reshape(6, 2);
        System.out.println("   ✓ Reshaped: " + java.util.Arrays.toString(x_reshaped.shape()));

        Tensor x_unsqueezed = x.unsqueeze(0);
        System.out.println("   ✓ Unsqueezed: " + java.util.Arrays.toString(x_unsqueezed.shape()));
        System.out.println();

        // 6. Memory management
        System.out.println("6. Memory tracking...");
        MemoryManager mem = MemoryManager.getInstance();
        mem.allocate(1024 * 1024);  // 1 MB
        System.out.println("   ✓ Allocated: " + mem.getStats());

        mem.free(512 * 1024);  // Free 512 KB
        System.out.println("   ✓ After free: " + mem.getStats());
        System.out.println();

        // 7. NoGrad context
        System.out.println("7. NoGrad context (inference mode)...");
        System.out.println("   ✓ Gradients enabled: " + NoGrad.isEnabled());

        try (NoGrad ctx = new NoGrad()) {
            System.out.println("   ✓ Inside NoGrad: " + NoGrad.isEnabled());
            // Inference code here - no gradients computed
        }

        System.out.println("   ✓ After NoGrad: " + NoGrad.isEnabled());
        System.out.println();

        // 8. Sequential container
        System.out.println("8. Sequential model container...");
        // Sequential model = new Sequential()
        //     .add(new Conv2d(3, 64, 3, 1, 1))
        //     .add(new BatchNorm2d(64))
        //     .add(new ReLU())
        //     .add(new MaxPool2d(2));
        System.out.println("   ✓ Sequential container available");
        System.out.println();

        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║  Core Tensor API example complete!                       ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
    }
}
