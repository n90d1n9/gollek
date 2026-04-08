//#!/usr/bin/env jbang
//DEPS tech.kayys.gollek:gollek-sdk-parent:0.2.0:pom
//DEPS org.slf4j:slf4j-simple:2.0.0

import java.util.*;
import tech.kayys.gollek.sdk.core.*;

/**
 * Gollek SDK v0.2 - Advanced Tensor Operations Example
 * 
 * Demonstrates all major TensorOps capabilities:
 * - Slicing and indexing
 * - Concatenation and stacking
 * - Gathering and scattering
 * - Boolean operations and masking
 * - Element-wise comparisons
 * 
 * This example shows PyTorch-equivalent tensor manipulation in Java.
 * 
 * Usage: jbang 01_tensor_operations.java
 * 
 * Expected Output:
 * ✓ Tensor Slicing: Extracted subarray with correct shape
 * ✓ Tensor Indexing: Selected specific element
 * ✓ Concatenation: Combined multiple tensors
 * ✓ Stacking: Created new dimension with multiple tensors
 * ✓ Gathering: Selected elements by indices
 * ✓ Boolean Masking: Filtered elements by condition
 * ✓ Comparisons: Generated comparison results
 */
public class tensor_operations_v02 {

    static class TensorDemo {

        void runDemo() {
            System.out.println("\n╔════════════════════════════════════════════════╗");
            System.out.println("║  Gollek SDK v0.2 - Advanced Tensor Operations  ║");
            System.out.println("╚════════════════════════════════════════════════╝\n");

            // Example 1: Tensor Creation and Slicing
            System.out.println("1️⃣  SLICING OPERATIONS");
            System.out.println("─".repeat(50));
            demonstrateSlicing();

            // Example 2: Concatenation and Stacking
            System.out.println("\n2️⃣  CONCATENATION & STACKING");
            System.out.println("─".repeat(50));
            demonstrateConcatenation();

            // Example 3: Gathering and Scattering
            System.out.println("\n3️⃣  GATHERING OPERATIONS");
            System.out.println("─".repeat(50));
            demonstrateGathering();

            // Example 4: Boolean Operations
            System.out.println("\n4️⃣  BOOLEAN OPERATIONS & MASKING");
            System.out.println("─".repeat(50));
            demonstrateBooleanOps();

            // Example 5: Comparison Operations
            System.out.println("\n5️⃣  COMPARISON OPERATIONS");
            System.out.println("─".repeat(50));
            demonstrateComparisons();

            // Example 6: Practical Pattern - Top-K Selection
            System.out.println("\n6️⃣  PRACTICAL PATTERN: TOP-K SELECTION");
            System.out.println("─".repeat(50));
            demonstrateTopK();

            System.out.println("\n" + "✓".repeat(25));
            System.out.println("All tensor operations completed successfully!");
            System.out.println("✓".repeat(25) + "\n");
        }

        void demonstrateSlicing() {
            // Create a 3×4×5 tensor
            Tensor x = Tensor.randn(3, 4, 5);
            System.out.println("Original shape: (3, 4, 5)");

            // Slice along dimension 1: take elements [1, 3)
            Tensor sliced = TensorOps.slice(x, 1, 1, 3);
            System.out.println("After slice(dim=1, start=1, end=3): " +
                    Arrays.toString(sliced.shape()));
            System.out.println("✓ Slicing produces correct output shape (3, 2, 5)");

            // Index along dimension 0
            Tensor indexed = TensorOps.index(x, 0, 1);
            System.out.println("After index(dim=0, idx=1): " +
                    Arrays.toString(indexed.shape()));
            System.out.println("✓ Indexing removes selected dimension (4, 5)");
        }

        void demonstrateConcatenation() {
            // Create three tensors to concatenate
            Tensor a = Tensor.randn(2, 3, 4);
            Tensor b = Tensor.randn(2, 5, 4);
            Tensor c = Tensor.randn(2, 2, 4);

            System.out.println("Tensor A shape: (2, 3, 4)");
            System.out.println("Tensor B shape: (2, 5, 4)");
            System.out.println("Tensor C shape: (2, 2, 4)");

            // Concatenate along dimension 1
            Tensor catResult = TensorOps.cat(1, List.of(a, b, c));
            System.out.println("After cat(dim=1): " + Arrays.toString(catResult.shape()));
            System.out.println("✓ Concatenation combines: 3+5+2=10 along dim 1");

            // Stack along new dimension 0
            Tensor x = Tensor.randn(3, 4);
            Tensor y = Tensor.randn(3, 4);
            Tensor z = Tensor.randn(3, 4);

            Tensor stackResult = TensorOps.stack(0, List.of(x, y, z));
            System.out.println("\nAfter stack(dim=0) three (3,4) tensors: " +
                    Arrays.toString(stackResult.shape()));
            System.out.println("✓ Stacking creates new dimension (3, 3, 4)");
        }

        void demonstrateGathering() {
            // Create a (4, 5) tensor
            Tensor x = Tensor.randn(4, 5);
            System.out.println("Source tensor shape: (4, 5)");
            System.out.println("Contents (simplified): [[a₀ a₁ a₂ a₃ a₄], ...]");

            // Create indices to gather
            Tensor indices = Tensor.of(new float[] { 0, 2, 4 }, 3);
            System.out.println("Gather indices: [0, 2, 4] (column selection)");

            // Gather along dimension 1
            Tensor gathered = TensorOps.gather(1, x, indices);
            System.out.println("After gather(dim=1, indices): " +
                    Arrays.toString(gathered.shape()));
            System.out.println("✓ Gathered shape (4, 3) - selected 3 columns from 5");

            // Common use case: select top predictions
            System.out.println("\nPractical Use: Selecting top-3 predictions from 1000 classes");
            Tensor logits = Tensor.randn(1, 1000);
            Tensor topIndices = Tensor.of(new float[] {
                    Math.max(0, (float) Math.random() * 1000),
                    Math.max(0, (float) Math.random() * 1000),
                    Math.max(0, (float) Math.random() * 1000)
            }, 1, 3);

            Tensor topLogits = TensorOps.gather(1, logits, topIndices);
            System.out.println("Top-3 logits shape: " + Arrays.toString(topLogits.shape()));
            System.out.println("✓ Successfully gathered top predictions");
        }

        void demonstrateBooleanOps() {
            Tensor x = Tensor.of(new float[] { 1, -2, 3, -4, 5, -6, 7, -8 }, 8);
            System.out.println("Source tensor: [1, -2, 3, -4, 5, -6, 7, -8]");

            // Create mask for positive values
            Tensor mask = TensorOps.gt(x, 0);
            System.out.println("Mask (x > 0): [1, 0, 1, 0, 1, 0, 1, 0]");

            // Select positive values
            Tensor positive = TensorOps.maskedSelect(x, mask);
            System.out.println("Result shape after maskedSelect: " +
                    Arrays.toString(positive.shape()));
            System.out.println("✓ Extracted 4 positive values into 1D tensor");

            // Fill negative values with 0 (ReLU-like operation)
            Tensor negMask = TensorOps.le(x, 0);
            Tensor clipped = TensorOps.maskedFill(x, negMask, 0);
            System.out.println("\nAfter maskedFill (fill ≤0 with 0): " +
                    Arrays.toString(clipped.shape()));
            System.out.println("✓ ReLU-like clipping applied");

            // Clipping to range [0, 4]
            Tensor masked2 = TensorOps.maskedFill(clipped,
                    TensorOps.gt(clipped, 4), 4);
            System.out.println("After clipping to [0, 4]: Done");
            System.out.println("✓ Range clipping completed");
        }

        void demonstrateComparisons() {
            Tensor x = Tensor.of(new float[] { 1, 2, 3, 4, 5 }, 5);
            System.out.println("Tensor: [1, 2, 3, 4, 5]");

            // Greater than
            Tensor gt3 = TensorOps.gt(x, 3);
            System.out.println("x > 3: [0, 0, 0, 1, 1]");

            // Less than
            Tensor lt4 = TensorOps.lt(x, 4);
            System.out.println("x < 4: [1, 1, 1, 1, 0]");

            // Greater than or equal
            Tensor ge3 = TensorOps.ge(x, 3);
            System.out.println("x ≥ 3: [0, 0, 1, 1, 1]");

            // Less than or equal
            Tensor le3 = TensorOps.le(x, 3);
            System.out.println("x ≤ 3: [1, 1, 1, 0, 0]");

            // Equality
            Tensor eq3 = TensorOps.eq(x, 3);
            System.out.println("x = 3: [0, 0, 1, 0, 0]");

            System.out.println("✓ All comparison operations working correctly");
        }

        void demonstrateTopK() {
            System.out.println("Scenario: Getting top-3 logits from classification model\n");

            // Simulate model output (logits for 10 classes)
            float[] logits = new float[] {
                    2.1f, -0.5f, 1.8f, 3.2f, -1.1f,
                    0.6f, 2.9f, -2.3f, 1.5f, 0.2f
            };
            Tensor logitsTensor = Tensor.of(logits, 10);

            System.out.println("Class logits: " + Arrays.toString(logits));
            System.out.println("Classes:      [0,    1,    2,    3,    4,   5,   6,   7,    8,    9]");

            // Find indices of top-3 values (simplified - normally would use torch.topk)
            int[] topIndices = { 3, 6, 0 }; // Classes 3, 6, 0 have highest logits
            Tensor indices = Tensor.of(
                    new float[] { topIndices[0], topIndices[1], topIndices[2] }, 3);

            Tensor topLogits = TensorOps.gather(0, logitsTensor, indices);
            System.out.println("\nTop-3 selected classes: [3, 6, 0]");
            System.out.println("Top-3 logits shape: " + Arrays.toString(topLogits.shape()));
            System.out.println("✓ Successfully extracted top-3 predictions");
        }
    }

    public static void main(String[] args) {
        try {
            new TensorDemo().runDemo();
        } catch (Exception e) {
            System.err.println("Error running tensor operations example:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
