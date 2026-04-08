//DEPS tech.kayys.gollek:gollek-sdk-nn:0.1.0-SNAPSHOT
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.GradTensor;

/**
 * Advanced example: Creating a custom Module structure.
 * This demonstrates how to build modular, reusable architectures beyond Sequential.
 */
class ResidualBlock extends Module {
    private final Linear fc1;
    private final Linear fc2;
    private final ReLU relu = new ReLU();

    public ResidualBlock(int dim) {
        this.fc1 = register("fc1", new Linear(dim, dim));
        this.fc2 = register("fc2", new Linear(dim, dim));
    }

    @Override
    public GradTensor forward(GradTensor input) {
        // x = ReLU(fc2(ReLU(fc1(input))) + input)
        GradTensor out = relu.forward(fc1.forward(input));
        out = fc2.forward(out);
        return relu.forward(out.add(input)); // Residual connection
    }
}

public class custom_module_demo {
    public static void main(String[] args) {
        System.out.println("=== Custom Module Architecture ===");

        // Building a model using our custom block
        Sequential model = new Sequential(
            new Linear(64, 64),
            new ResidualBlock(64),
            new ResidualBlock(64),
            new Linear(64, 10)
        );

        System.out.println("Model with residual connections created:");
        System.out.println(model);
        
        GradTensor input = GradTensor.randn(1, 64);
        GradTensor output = model.forward(input);
        
        System.out.println("Output shape: " + java.util.Arrays.toString(output.shape()));
    }
}
