///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek SDK Data Augmentation Example
//
// Demonstrates data augmentation pipeline with:
// - Multiple augmentation transforms
// - Composing augmentations
// - Training vs inference modes
//
// Prerequisites:
//   - Java 25+
//   - JBang: curl -Ls https://sh.jbang.dev | bash -s -
//   - All SDK modules built
//
// Usage:
//   jbang gollek-sdk-augment-example.java
//
// DEPS tech.kayys.gollek:gollek-sdk-augment:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-sdk-vision:0.1.0-SNAPSHOT
// JAVA 25+

import tech.kayys.gollek.sdk.augment.*;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.util.Random;

public class gollek_sdk_augment_example {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║     Gollek SDK Data Augmentation Example                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // 1. Create individual augmentations
        System.out.println("1. Creating individual augmentations...");

        Augmentation flip = Augmentation.RandomHorizontalFlip.create();
        Augmentation crop = Augmentation.RandomCrop.of(224);
        Augmentation color = Augmentation.ColorJitter.of(0.2, 0.2, 0.2);
        Augmentation erase = Augmentation.RandomErasing.of(0.25);
        Augmentation rotate = Augmentation.RandomRotation.of(15.0);
        Augmentation grayscale = Augmentation.RandomGrayscale.of(0.1);
        Augmentation blur = Augmentation.GaussianBlur.of(5, 1.0);

        System.out.println("   ✓ RandomHorizontalFlip (p=0.5)");
        System.out.println("   ✓ RandomCrop (size=224)");
        System.out.println("   ✓ ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)");
        System.out.println("   ✓ RandomErasing (p=0.25)");
        System.out.println("   ✓ RandomRotation (max_angle=15°)");
        System.out.println("   ✓ RandomGrayscale (p=0.1)");
        System.out.println("   ✓ GaussianBlur (kernel=5, sigma=1.0)");
        System.out.println();

        // 2. Create augmentation pipeline
        System.out.println("2. Creating augmentation pipeline...");

        AugmentationPipeline trainAugment = new AugmentationPipeline(
            flip,
            crop,
            color,
            erase
        );

        System.out.println("   ✓ Training pipeline:");
        System.out.println("     - RandomHorizontalFlip");
        System.out.println("     - RandomCrop");
        System.out.println("     - ColorJitter");
        System.out.println("     - RandomErasing");
        System.out.println();

        // 3. Create inference pipeline (minimal augmentations)
        System.out.println("3. Creating inference pipeline...");

        AugmentationPipeline inferAugment = new AugmentationPipeline(
            Augmentation.RandomHorizontalFlip.of(0.0)  // No flip during inference
        );

        System.out.println("   ✓ Inference pipeline:");
        System.out.println("     - No augmentations (deterministic)");
        System.out.println();

        // 4. Apply augmentations
        System.out.println("4. Applying augmentations to sample data...");

        Random rng = new Random(42);
        GradTensor input = GradTensor.randn(3, 256, 256);  // Simulated image

        System.out.println("   ✓ Input shape: " + java.util.Arrays.toString(input.shape()));

        GradTensor trainOutput = trainAugment.apply(input);
        System.out.println("   ✓ Training output shape: " + java.util.Arrays.toString(trainOutput.shape()));

        GradTensor inferOutput = inferAugment.apply(input);
        System.out.println("   ✓ Inference output shape: " + java.util.Arrays.toString(inferOutput.shape()));
        System.out.println();

        // 5. Show augmentation effects
        System.out.println("5. Augmentation statistics...");
        System.out.println("   Training pipeline:");
        System.out.println("     - " + trainAugment.size() + " transforms");
        System.out.println("     - Stochastic (random)");
        System.out.println("     - Improves generalization");
        System.out.println();
        System.out.println("   Inference pipeline:");
        System.out.println("     - " + inferAugment.size() + " transforms");
        System.out.println("     - Deterministic");
        System.out.println("     - Consistent predictions");
        System.out.println();

        // 6. Common augmentation recipes
        System.out.println("6. Common augmentation recipes...");

        System.out.println("   a) ImageNet training:");
        System.out.println("      RandomResizedCrop(224) → RandomHorizontalFlip() → ColorJitter() → Normalize()");

        System.out.println("   b) CIFAR-10 training:");
        System.out.println("      RandomCrop(32, padding=4) → RandomHorizontalFlip() → Normalize()");

        System.out.println("   c) Medical imaging:");
        System.out.println("      RandomRotation(15) → RandomVerticalFlip() → Normalize()");

        System.out.println("   d) Satellite imagery:");
        System.out.println("      RandomRotation(360) → RandomHorizontalFlip() → RandomVerticalFlip()");
        System.out.println();

        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║  Data Augmentation example complete!                     ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Tip: Use different pipelines for training vs inference!");
        System.out.println("  - Training: Heavy augmentation for generalization");
        System.out.println("  - Inference: Minimal/No augmentation for consistency");
    }
}
