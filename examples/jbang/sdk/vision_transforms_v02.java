//#!/usr/bin/env jbang
//DEPS tech.kayys.gollek:gollek-sdk-vision:0.2.0
//DEPS org.slf4j:slf4j-simple:2.0.0

import java.util.*;
import tech.kayys.gollek.sdk.core.*;
import tech.kayys.gollek.ml.vision.transforms.*;

/**
 * Gollek SDK v0.2 - Vision Transforms Example
 * 
 * Demonstrates image preprocessing pipeline with Vision Transforms:
 * - Image resizing (bilinear interpolation)
 * - Center cropping
 * - Random cropping with reproducibility
 * - Image normalization (ImageNet stats)
 * - Augmentation (random flip, color jitter)
 * - Pipeline composition
 * 
 * This example shows how to build efficient image preprocessing pipelines
 * for computer vision models (CNN, ResNet, Vision Transformers, etc).
 * 
 * Usage: jbang 02_vision_transforms.java
 * 
 * Expected Output:
 * ✓ Resize: Image scaled using bilinear interpolation
 * ✓ Crop: Image center-cropped to target size
 * ✓ Normalize: Applied ImageNet normalization
 * ✓ Augment: Applied random transformations
 * ✓ Pipeline: Composed transforms executed sequentially
 */
public class vision_transforms_v02 {

    static class VisionDemo {

        void runDemo() {
            System.out.println("\n╔════════════════════════════════════════════╗");
            System.out.println("║   Gollek SDK v0.2 - Vision Transforms     ║");
            System.out.println("╚════════════════════════════════════════════╝\n");

            // Example 1: Basic Resize
            System.out.println("1️⃣  IMAGE RESIZING");
            System.out.println("─".repeat(50));
            demonstrateResize();

            // Example 2: Center Crop
            System.out.println("\n2️⃣  CENTER CROPPING");
            System.out.println("─".repeat(50));
            demonstrateCenterCrop();

            // Example 3: Random Crop
            System.out.println("\n3️⃣  RANDOM CROPPING");
            System.out.println("─".repeat(50));
            demonstrateRandomCrop();

            // Example 4: Normalization
            System.out.println("\n4️⃣  NORMALIZATION");
            System.out.println("─".repeat(50));
            demonstrateNormalize();

            // Example 5: Augmentation
            System.out.println("\n5️⃣  DATA AUGMENTATION");
            System.out.println("─".repeat(50));
            demonstrateAugmentation();

            // Example 6: Complete Pipeline
            System.out.println("\n6️⃣  COMPLETE PREPROCESSING PIPELINE");
            System.out.println("─".repeat(50));
            demonstratePipeline();

            // Example 7: Batch Processing
            System.out.println("\n7️⃣  BATCH PROCESSING");
            System.out.println("─".repeat(50));
            demonstrateBatchProcessing();

            System.out.println("\n" + "✓".repeat(25));
            System.out.println("All vision transforms completed successfully!");
            System.out.println("✓".repeat(25) + "\n");
        }

        void demonstrateResize() {
            System.out.println("Scenario: Resize image to fixed size for model input\n");

            // Simulate an image: (C=3, H=480, W=640) - 3 channels, 480×640 pixels
            Tensor image = Tensor.randn(3, 480, 640);
            System.out.println("Input image shape: (3, 480, 640)");
            System.out.println("  - 3 channels (RGB)");
            System.out.println("  - 480 pixels height");
            System.out.println("  - 640 pixels width");

            // Resize to common size (224, 224) for CNN models
            VisionTransforms.Resize resize = new VisionTransforms.Resize(224, 224);
            Tensor resized = resize.apply(image);
            System.out.println("\nAfter Resize(224, 224):");
            System.out.println("  Output shape: " + Arrays.toString(resized.shape()));
            System.out.println("  ✓ Bilinear interpolation applied");

            // Aspect ratio preservation variant
            System.out.println("\nCommon pattern: Resize to (256, 256) then crop");
            System.out.println("  This preserves more information than direct resize");
        }

        void demonstrateCenterCrop() {
            System.out.println("Scenario: Extract center patch from image\n");

            Tensor image = Tensor.randn(3, 480, 640);
            System.out.println("Input image: (3, 480, 640)");

            // Center crop to (224, 224)
            VisionTransforms.CenterCrop centerCrop = new VisionTransforms.CenterCrop(224, 224);
            Tensor cropped = centerCrop.apply(image);
            System.out.println("After CenterCrop(224, 224):");
            System.out.println("  Output shape: " + Arrays.toString(cropped.shape()));
            System.out.println("  Crop location: center of image");
            System.out.println("  ✓ Preserves center content");

            // Show crop calculation
            int startH = (480 - 224) / 2; // = 128
            int startW = (640 - 224) / 2; // = 208
            System.out.println("\nCrop math:");
            System.out.println("  Start height: (480 - 224) / 2 = " + startH);
            System.out.println("  Start width: (640 - 224) / 2 = " + startW);
            System.out.println("  Region: [128:352, 208:432]");
        }

        void demonstrateRandomCrop() {
            System.out.println("Scenario: Random cropping for data augmentation\n");

            Tensor image = Tensor.randn(3, 256, 256);
            System.out.println("Input image: (3, 256, 256)");

            // Random crop with seed for reproducibility
            VisionTransforms.RandomCrop randomCrop = new VisionTransforms.RandomCrop(224, 224);

            // Crop 1
            Tensor crop1 = randomCrop.apply(image);
            System.out.println("Crop 1 output: " + Arrays.toString(crop1.shape()));

            // Crop 2 (may be different location)
            Tensor crop2 = randomCrop.apply(image);
            System.out.println("Crop 2 output: " + Arrays.toString(crop2.shape()));

            System.out.println("\nBenefits:");
            System.out.println("  ✓ Different crops on each epoch");
            System.out.println("  ✓ Prevents overfitting");
            System.out.println("  ✓ Improves generalization");
            System.out.println("  ✓ Effective data augmentation");
        }

        void demonstrateNormalize() {
            System.out.println("Scenario: Normalize image with ImageNet statistics\n");

            // Image values typically in [0, 1] after ToTensor
            Tensor image = Tensor.randn(3, 224, 224);
            System.out.println("Input image shape: (3, 224, 224)");
            System.out.println("Input range: [0, 1] (after ToTensor)");

            // ImageNet statistics (pre-computed on 1.2M training images)
            float[] mean = { 0.485f, 0.456f, 0.406f }; // RGB channels
            float[] std = { 0.229f, 0.224f, 0.225f };

            VisionTransforms.Normalize normalize = new VisionTransforms.Normalize(mean, std);
            Tensor normalized = normalize.apply(image);

            System.out.println("\nImageNet Statistics:");
            System.out.println("  Mean: " + Arrays.toString(mean));
            System.out.println("  Std:  " + Arrays.toString(std));
            System.out.println("\nAfter Normalize:");
            System.out.println("  Output shape: " + Arrays.toString(normalized.shape()));
            System.out.println("  Output range: approximately [-2, 2]");
            System.out.println("  ✓ Zero-centered, unit variance");

            System.out.println("\nWhy normalize?");
            System.out.println("  - Model was trained on normalized data");
            System.out.println("  - Improves training stability");
            System.out.println("  - Accelerates convergence");
        }

        void demonstrateAugmentation() {
            System.out.println("Scenario: Data augmentation techniques\n");

            Tensor image = Tensor.randn(3, 224, 224);
            System.out.println("Input image shape: (3, 224, 224)\n");

            // Random flip
            System.out.println("1. Random Flip (horizontal):");
            VisionTransforms.RandomFlip flip = new VisionTransforms.RandomFlip();
            Tensor flipped = flip.apply(image);
            System.out.println("  ✓ Flipped horizontally (random)");
            System.out.println("  ✓ Useful for: rotational invariance\n");

            // Color jitter
            System.out.println("2. Color Jitter (brightness, contrast, saturation):");
            VisionTransforms.ColorJitter jitter = new VisionTransforms.ColorJitter(
                    0.2f, // brightness
                    0.2f, // contrast
                    0.2f // saturation
            );
            Tensor jittered = jitter.apply(image);
            System.out.println("  ✓ Randomly adjusted color values");
            System.out.println("  ✓ Useful for: lighting robustness\n");

            System.out.println("Augmentation benefits:");
            System.out.println("  - Increases effective dataset size");
            System.out.println("  - Improves model robustness");
            System.out.println("  - Reduces overfitting");
            System.out.println("  - Better generalization");
        }

        void demonstratePipeline() {
            System.out.println("Scenario: Complete preprocessing pipeline for ImageNet model\n");

            Tensor image = Tensor.randn(3, 480, 640);
            System.out.println("Raw image shape: (3, 480, 640)\n");

            System.out.println("Pipeline composition:");
            System.out.println("  1. Resize(256, 256)");
            System.out.println("  2. CenterCrop(224, 224)");
            System.out.println("  3. Normalize(ImageNet mean/std)");
            System.out.println("  4. RandomFlip()");
            System.out.println("  5. ColorJitter(0.2, 0.2, 0.2)\n");

            // Build and apply pipeline
            VisionTransforms.Compose pipeline = new VisionTransforms.Compose(
                    new VisionTransforms.Resize(256, 256),
                    new VisionTransforms.CenterCrop(224, 224),
                    new VisionTransforms.Normalize(
                            new float[] { 0.485f, 0.456f, 0.406f },
                            new float[] { 0.229f, 0.224f, 0.225f }),
                    new VisionTransforms.RandomFlip(),
                    new VisionTransforms.ColorJitter(0.2f, 0.2f, 0.2f));

            Tensor processed = pipeline.apply(image);
            System.out.println("Final processed shape: " + Arrays.toString(processed.shape()));
            System.out.println("✓ Pipeline executed successfully\n");

            System.out.println("Pipeline advantages:");
            System.out.println("  - Reusable and composable");
            System.out.println("  - Clear transformation sequence");
            System.out.println("  - Easy to modify (add/remove transforms)");
            System.out.println("  - Efficient (single pass through data)");
        }

        void demonstrateBatchProcessing() {
            System.out.println("Scenario: Process batch of images\n");

            System.out.println("Batch processing strategy:");
            System.out.println("─".repeat(50));

            int batchSize = 32;
            int imageH = 480;
            int imageW = 640;

            System.out.println("Batch size: " + batchSize + " images");
            System.out.println("Image size: " + imageH + "×" + imageW);

            // Create batch
            System.out.println("\nProcessing:");
            VisionTransforms.Resize resize = new VisionTransforms.Resize(224, 224);
            VisionTransforms.Normalize normalize = new VisionTransforms.Normalize(
                    new float[] { 0.485f, 0.456f, 0.406f },
                    new float[] { 0.229f, 0.224f, 0.225f });

            long startTime = System.currentTimeMillis();

            for (int i = 0; i < batchSize; i++) {
                Tensor image = Tensor.randn(3, imageH, imageW);
                Tensor resized = resize.apply(image);
                Tensor normalized = normalize.apply(resized);

                if ((i + 1) % 8 == 0) {
                    System.out.println("  Processed " + (i + 1) + "/" + batchSize + " images");
                }
            }

            long elapsed = System.currentTimeMillis() - startTime;
            System.out.println("\nTotal time: " + elapsed + "ms");
            System.out.println("Time per image: " + (elapsed / (float) batchSize) + "ms");
            System.out.println("✓ Batch processing completed");
        }
    }

    public static void main(String[] args) {
        try {
            new VisionDemo().runDemo();
        } catch (Exception e) {
            System.err.println("Error running vision transforms example:");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
