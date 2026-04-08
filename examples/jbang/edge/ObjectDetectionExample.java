///usr/bin/env jbang "$0" "$@" ; exit $?
// Enhanced Gollek LiteRT Object Detection Example
//
// DEPENDENCY RESOLUTION:
// The SDK modules must be built and installed to local Maven repository first:
//   cd gollek && mvn clean install -pl sdk/lib/gollek-sdk-litert -am -DskipTests
//
// REPOS mavenLocal
// DEPS tech.kayys.gollek:gollek-sdk-litert:0.1.0-SNAPSHOT
// DEPS tech.kayys.gollek:gollek-runner-litert:0.1.0-SNAPSHOT
// JAVA 21+
//
// Usage:
//   jbang ObjectDetectionExample.java --model ssd_mobilenet.litert --image street.jpg
//   jbang ObjectDetectionExample.java --model model.litert --image street.jpg --threshold 0.6
//   jbang ObjectDetectionExample.java --model model.litert --image street.jpg --delegate GPU

package tech.kayys.gollek.examples.edge;

import tech.kayys.gollek.sdk.litert.LiteRTSdk;
import tech.kayys.gollek.sdk.litert.config.LiteRTConfig;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * LiteRT Object Detection Example
 * 
 * Demonstrates object detection using LiteRT models like SSD MobileNet, YOLO, etc.
 * 
 * Usage:
 *   jbang ObjectDetectionExample.java --model ssd_mobilenet.litert --image street.jpg
 *   jbang ObjectDetectionExample.java --model ssd_mobilenet.litert --image street.jpg --threshold 0.5
 *   jbang ObjectDetectionExample.java --model ssd_mobilenet.litert --image street.jpg --delegate GPU
 * 
 * @author Gollek Team
 * @version 0.1.0
 */
public class ObjectDetectionExample {

    // Detection output format
    record Detection(
        int classId,
        float confidence,
        float x,
        float y,
        float width,
        float height
    ) {
        @Override
        public String toString() {
            return String.format("Class %d (%.2f%%) at [%.0f, %.0f, %.0f, %.0f]",
                classId, confidence * 100, x, y, width, height);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║          LiteRT Object Detection Example                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        // Parse arguments
        String modelPath = System.getProperty("model", "ssd_mobilenet.litert");
        String imagePath = System.getProperty("image", null);
        float threshold = Float.parseFloat(System.getProperty("threshold", "0.5"));
        String delegateStr = System.getProperty("delegate", "AUTO");
        String outputPath = System.getProperty("output", null);

        if (imagePath == null) {
            System.err.println("Error: --image parameter is required");
            System.err.println("Usage: jbang ObjectDetectionExample.java --model <model.litert> --image <image.jpg>");
            System.exit(1);
        }

        LiteRTConfig.Delegate delegate = LiteRTConfig.Delegate.valueOf(delegateStr.toUpperCase());

        // Load image
        System.out.println("Loading image: " + imagePath);
        BufferedImage image = ImageIO.read(new File(imagePath));
        if (image == null) {
            System.err.println("Error: Could not load image: " + imagePath);
            System.exit(1);
        }
        System.out.println("  Size: " + image.getWidth() + "x" + image.getHeight());
        System.out.println();

        // Create SDK
        LiteRTConfig config = LiteRTConfig.builder()
            .numThreads(4)
            .delegate(delegate)
            .enableXnnpack(true)
            .useMemoryPool(true)
            .build();

        try (LiteRTSdk sdk = new LiteRTSdk(config)) {
            // Load model
            System.out.println("Loading model: " + modelPath);
            sdk.loadModel("detector", Path.of(modelPath));
            System.out.println("✓ Model loaded");
            System.out.println();

            // Preprocess image
            System.out.println("Preprocessing image...");
            byte[] inputData = preprocessImage(image, 300, 300);
            System.out.println("  Input size: " + inputData.length + " bytes");
            System.out.println();

            // Run inference
            System.out.println("Running object detection...");
            long startTime = System.currentTimeMillis();
            
            InferenceRequest request = InferenceRequest.builder()
                .model("detector")
                .inputData(inputData)
                .build();

            InferenceResponse response = sdk.infer(request);
            long elapsed = System.currentTimeMillis() - startTime;

            System.out.println("✓ Detection completed in " + elapsed + "ms");
            System.out.println();

            // Parse detections
            System.out.println("Parsing detections...");
            java.util.List<Detection> detections = parseDetections(response.getOutputData(), threshold);
            System.out.println("  Found " + detections.size() + " objects above threshold " + threshold);
            System.out.println();

            // Print detections
            if (!detections.isEmpty()) {
                System.out.println("Detections:");
                for (int i = 0; i < detections.size(); i++) {
                    Detection det = detections.get(i);
                    System.out.printf("  %d. %s%n", i + 1, det);
                }
                System.out.println();
            }

            // Draw detections on image
            System.out.println("Drawing detections on image...");
            BufferedImage outputImage = drawDetections(image, detections);
            
            if (outputPath != null) {
                ImageIO.write(outputImage, "jpg", new File(outputPath));
                System.out.println("  Saved to: " + outputPath);
            } else {
                String defaultOutput = imagePath.replaceFirst("\\.\\w+$", "_detected.jpg");
                ImageIO.write(outputImage, "jpg", new File(defaultOutput));
                System.out.println("  Saved to: " + defaultOutput);
            }
            System.out.println();

            // Run batch detection with different thresholds
            System.out.println("Running batch detection (multiple thresholds)...");
            batchDetection(sdk, inputData, threshold);

            // Display metrics
            displayMetrics(sdk);
        }

        System.out.println();
        System.out.println("✓ Example completed successfully");
    }

    /**
     * Preprocess image for SSD MobileNet input.
     */
    private static byte[] preprocessImage(BufferedImage image, int targetWidth, int targetHeight) {
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(image, 0, 0, targetWidth, targetHeight, null);

        // Convert to float32 tensor [1, height, width, 3]
        ByteBuffer buffer = ByteBuffer.allocate(1 * targetHeight * targetWidth * 3 * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // Normalize to [0, 1] for SSD
                buffer.putFloat(r / 255.0f);
                buffer.putFloat(g / 255.0f);
                buffer.putFloat(b / 255.0f);
            }
        }

        return buffer.array();
    }

    /**
     * Parse detection outputs.
     * SSD MobileNet output format: [1, num_detections, 6]
     * Each detection: [batch_id, class_id, confidence, x, y, width, height]
     */
    private static java.util.List<Detection> parseDetections(byte[] outputData, float threshold) {
        ByteBuffer buffer = ByteBuffer.wrap(outputData);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        java.util.List<Detection> detections = new java.util.ArrayList<>();

        // Parse detections (adjust based on your model's output format)
        int numDetections = buffer.capacity() / (6 * 4); // 6 floats per detection

        for (int i = 0; i < numDetections; i++) {
            int classId = (int) buffer.getFloat();
            float confidence = buffer.getFloat();
            float x = buffer.getFloat();
            float y = buffer.getFloat();
            float width = buffer.getFloat();
            float height = buffer.getFloat();

            if (confidence >= threshold) {
                detections.add(new Detection(classId, confidence, x, y, width, height));
            }
        }

        // Sort by confidence
        detections.sort((a, b) -> Float.compare(b.confidence, a.confidence));

        return detections;
    }

    /**
     * Draw detections on image.
     */
    private static BufferedImage drawDetections(BufferedImage image, java.util.List<Detection> detections) {
        BufferedImage output = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = output.createGraphics();
        g.drawImage(image, 0, 0, null);

        // Set drawing properties
        g.setStroke(new BasicStroke(3));
        g.setFont(new Font("Arial", Font.BOLD, 16));

        // Color palette for different classes
        Color[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.ORANGE, Color.MAGENTA, Color.CYAN};

        for (Detection det : detections) {
            Color color = colors[det.classId % colors.length];

            // Convert normalized coordinates to pixels
            int x = (int) (det.x * image.getWidth());
            int y = (int) (det.y * image.getHeight());
            int w = (int) (det.width * image.getWidth());
            int h = (int) (det.height * image.getHeight());

            // Draw bounding box
            g.setColor(color);
            g.drawRect(x, y, w, h);
            g.drawRect(x + 1, y + 1, w - 2, h - 2);

            // Draw label
            String label = String.format("%s %.0f%%", getLabel(det.classId), det.confidence * 100);
            g.setColor(color);
            g.fillRect(x, y - 20, g.getFontMetrics().stringWidth(label) + 10, 20);
            g.setColor(Color.WHITE);
            g.drawString(label, x + 5, y - 5);
        }

        g.dispose();
        return output;
    }

    /**
     * Run batch detection with different thresholds.
     */
    private static void batchDetection(LiteRTSdk sdk, byte[] inputData, float baseThreshold) throws Exception {
        float[] thresholds = {baseThreshold, 0.6f, 0.7f, 0.8f};
        
        java.util.List<InferenceRequest> requests = new java.util.ArrayList<>();
        for (float t : thresholds) {
            requests.add(InferenceRequest.builder()
                .model("detector")
                .inputData(inputData)
                .build());
        }

        long startTime = System.currentTimeMillis();
        java.util.List<InferenceResponse> responses = sdk.inferBatch(requests);
        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("  Processed " + responses.size() + " detections in " + elapsed + "ms");
        
        for (int i = 0; i < responses.size(); i++) {
            java.util.List<Detection> dets = parseDetections(responses.get(i).getOutputData(), thresholds[i]);
            System.out.println("    Threshold " + thresholds[i] + ": " + dets.size() + " objects");
        }
        System.out.println();
    }

    /**
     * Display performance metrics.
     */
    private static void displayMetrics(LiteRTSdk sdk) {
        System.out.println("Performance Metrics:");
        try {
            var metrics = sdk.getMetrics(null);
            System.out.println("  Total inferences:    " + metrics.getTotalInferences());
            System.out.println("  Avg latency:         " + String.format("%.2f", metrics.getAvgLatencyMs()) + " ms");
            System.out.println("  P95 latency:         " + String.format("%.2f", metrics.getP95LatencyMs()) + " ms");
            System.out.println("  Peak memory:         " + formatBytes(metrics.getPeakMemoryBytes()));
        } catch (Exception e) {
            System.out.println("  (Metrics not available)");
        }
    }

    /**
     * Get label for class index.
     */
    private static String getLabel(int classIndex) {
        // COCO dataset labels (simplified)
        String[] labels = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };
        
        if (classIndex >= 0 && classIndex < labels.length) {
            return labels[classIndex];
        }
        return "Class " + classIndex;
    }

    /**
     * Format bytes to human-readable string.
     */
    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0));
        return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    }
}
