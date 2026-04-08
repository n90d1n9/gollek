///usr/bin/env jbang "$0" "$@" ; exit $?
// Gollek SDK Vision Example - Image Classification
//
// Demonstrates computer vision workflow with:
// - ResNet model creation and forward pass
// - Image preprocessing with transforms
// - Prediction and class labels
//
// Prerequisites:
//   - Java 25+
//   - JBang: curl -Ls https://sh.jbang.dev | bash -s -
//   - All SDK modules built
//
// Usage:
//   jbang gollek-sdk-vision-example.java --demo
//   jbang gollek-sdk-vision-example.java --image cat.jpg
//
// DEPS tech.kayys.gollek:gollek-sdk-vision:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-sdk-core:0.1.0-SNAPSHOT
// JAVA 25+

import tech.kayys.gollek.sdk.vision.models.ResNet;
import tech.kayys.gollek.sdk.vision.transforms.Transform;
import tech.kayys.gollek.sdk.core.Tensor;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class gollek_sdk_vision_example {

    // ImageNet class labels (simplified - top 10)
    private static final String[] IMAGENET_CLASSES = {
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "cock", "hen", "ostrich"
    };

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║     Gollek SDK Vision Example - Image Classification     ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // 1. Create ResNet model
        System.out.println("1. Creating ResNet-18 model...");
        ResNet model = ResNet.resnet18(1000);
        System.out.println("   ✓ Model: " + model);
        System.out.println();

        // 2. Create preprocessing pipeline
        System.out.println("2. Creating preprocessing pipeline...");
        Transform preprocess = Transform.Compose.of(
            Transform.Resize.of(256),
            Transform.CenterCrop.of(224),
            Transform.ToTensor.of(),
            Transform.Normalize.imagenet()
        );
        System.out.println("   ✓ Pipeline: Resize(256) → CenterCrop(224) → ToTensor → Normalize");
        System.out.println();

        // 3. Load and preprocess image
        String imagePath = args.length > 0 ? args[0] : null;

        if (imagePath != null) {
            System.out.println("3. Loading image: " + imagePath);
            BufferedImage image = ImageIO.read(new File(imagePath));
            System.out.println("   ✓ Image loaded: " + image.getWidth() + "x" + image.getHeight());

            // Preprocess
            System.out.println("4. Preprocessing image...");
            GradTensor tensor = preprocess.apply(image);
            System.out.println("   ✓ Tensor shape: " + java.util.Arrays.toString(tensor.shape()));

            // Add batch dimension
            tensor = GradTensor.of(tensor.data(), 1, tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
            System.out.println("   ✓ Batch shape: [1, 3, 224, 224]");
            System.out.println();

            // 4. Run inference
            System.out.println("5. Running inference...");
            GradTensor logits = model.forward(tensor);
            System.out.println("   ✓ Logits shape: " + java.util.Arrays.toString(logits.shape()));
            System.out.println();

            // 5. Get prediction
            System.out.println("6. Getting prediction...");
            float[] logitsData = logits.data();
            int predictedClass = 0;
            float maxLogit = Float.NEGATIVE_INFINITY;

            for (int i = 0; i < logitsData.length; i++) {
                if (logitsData[i] > maxLogit) {
                    maxLogit = logitsData[i];
                    predictedClass = i;
                }
            }

            System.out.println("   ✓ Predicted class: " + predictedClass);
            System.out.println("   ✓ Confidence: " + String.format("%.2f", maxLogit));
            System.out.println();

            // 6. Show top-5 predictions
            System.out.println("7. Top-5 predictions:");
            float[] sorted = logitsData.clone();
            java.util.Arrays.sort(sorted);

            for (int i = 0; i < 5; i++) {
                float val = sorted[sorted.length - 1 - i];
                int idx = java.util.Arrays.binarySearch(logitsData, val);
                String label = idx < IMAGENET_CLASSES.length ? IMAGENET_CLASSES[idx] : "Class " + idx;
                System.out.printf("   %d. %-20s %.2f%n", i + 1, label, val);
            }
        } else {
            System.out.println("3. Running in demo mode (no image provided)...");
            System.out.println("   ✓ Creating random input tensor [1, 3, 224, 224]");

            GradTensor tensor = GradTensor.randn(1, 3, 224, 224);
            System.out.println("   ✓ Running forward pass...");

            GradTensor logits = model.forward(tensor);
            System.out.println("   ✓ Output shape: " + java.util.Arrays.toString(logits.shape()));
            System.out.println();

            System.out.println("To classify a real image:");
            System.out.println("  jbang gollek-sdk-vision-example.java --image cat.jpg");
        }

        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║  Vision example complete!                                ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
    }
}
