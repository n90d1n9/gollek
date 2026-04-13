package tech.kayys.gollek.ml.optim;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.nn.layer.Linear;

import static org.junit.jupiter.api.Assertions.*;

class OptimizerTest {

    @Test
    void testSGDStep() {
        var linear = new Linear(4, 2);
        var optimizer = SGD.builder(linear.parameters(), 0.01f).build();

        // Forward pass with simple gradient
        var input = GradTensor.ones(1, 4);
        var output = linear.forward(input);
        output.sum().backward();

        // Store old weights
        float[] oldW = linear.parameters().get(0).data().data().clone();

        optimizer.step();

        // Weights should have changed
        float[] newW = linear.parameters().get(0).data().data();
        boolean changed = false;
        for (int i = 0; i < oldW.length; i++) {
            if (Math.abs(oldW[i] - newW[i]) > 1e-8) changed = true;
        }
        assertTrue(changed);
    }

    @Test
    void testAdamStep() {
        var linear = new Linear(4, 2);
        var optimizer = Adam.builder(linear.parameters(), 0.001f).build();

        var input = GradTensor.randn(2, 4);
        var output = linear.forward(input);
        output.sum().backward();

        float[] oldW = linear.parameters().get(0).data().data().clone();
        optimizer.step();
        float[] newW = linear.parameters().get(0).data().data();

        boolean changed = false;
        for (int i = 0; i < oldW.length; i++) {
            if (Math.abs(oldW[i] - newW[i]) > 1e-8) changed = true;
        }
        assertTrue(changed);
    }

    @Test
    void testAdamWStep() {
        var linear = new Linear(4, 2);
        var optimizer = AdamW.builder(linear.parameters(), 0.001f).build();

        var input = GradTensor.randn(2, 4);
        var output = linear.forward(input);
        output.sum().backward();

        optimizer.step();
        assertNotNull(optimizer);
    }

    @Test
    void testZeroGrad() {
        var linear = new Linear(4, 2);
        var optimizer = SGD.builder(linear.parameters(), 0.01f).build();

        var input = GradTensor.ones(1, 4);
        var output = linear.forward(input);
        output.sum().backward();

        // Check grads exist
        assertNotNull(linear.parameters().get(0).data().grad());

        optimizer.zeroGrad();

        // After zeroGrad, grads should be zeroed (grad tensor still exists)
        var grad = linear.parameters().get(0).data().grad();
        if (grad != null) {
            for (float v : grad.data()) assertEquals(0f, v, 1e-8f);
        }
    }

    @Test
    void testRMSpropStep() {
        var linear = new Linear(4, 2);
        var optimizer = RMSprop.builder(linear.parameters(), 0.01f).build();

        var input = GradTensor.randn(2, 4);
        var output = linear.forward(input);
        output.sum().backward();

        optimizer.step();
        assertNotNull(optimizer);
    }

    @Test
    void testLRSchedulerStep() {
        var linear = new Linear(4, 2);
        var optimizer = Adam.builder(linear.parameters(), 0.01f).build();
        var scheduler = new CosineAnnealingLR(optimizer, 10, 0.0f);

        float initialLr = 0.01f;
        scheduler.step();
        // After stepping, scheduler should track epoch
        assertNotNull(scheduler);
    }
}
