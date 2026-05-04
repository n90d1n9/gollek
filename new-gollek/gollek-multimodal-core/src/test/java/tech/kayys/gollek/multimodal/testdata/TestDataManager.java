package tech.kayys.gollek.multimodal.testdata;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * Test data manager for multimodal testing.
 * Provides utilities for loading and managing test datasets.
 */
public class TestDataManager {

    private final Path testDataDir;
    private final Map<String, TestImage> imageCache = new HashMap<>();
    private final Map<String, TestDataset> datasetCache = new HashMap<>();

    public TestDataManager() {
        this.testDataDir = Path.of("src/test/resources");
    }

    public TestDataManager(Path testDataDir) {
        this.testDataDir = testDataDir;
    }

    /**
     * Load test image.
     */
    public TestImage loadImage(String filename) throws IOException {
        return imageCache.computeIfAbsent(filename, name -> {
            try {
                Path imagePath = testDataDir.resolve("images").resolve(name);
                if (!Files.exists(imagePath)) {
                    throw new RuntimeException("Image not found: " + name);
                }
                byte[] bytes = Files.readAllBytes(imagePath);
                return new TestImage(name, bytes, detectMimeType(name));
            } catch (IOException e) {
                throw new RuntimeException("Failed to load image: " + name, e);
            }
        });
    }

    /**
     * Load test document.
     */
    public TestDocument loadDocument(String filename) throws IOException {
        Path docPath = testDataDir.resolve("documents").resolve(filename);
        if (!Files.exists(docPath)) {
            throw new FileNotFoundException("Document not found: " + filename);
        }
        String content = Files.readString(docPath);
        return new TestDocument(filename, content, detectDocType(filename));
    }

    /**
     * Load test dataset.
     */
    public TestDataset loadDataset(String filename) throws IOException {
        return datasetCache.computeIfAbsent(filename, name -> {
            try {
                Path datasetPath = testDataDir.resolve("datasets").resolve(name);
                if (!Files.exists(datasetPath)) {
                    throw new RuntimeException("Dataset not found: " + name);
                }
                String json = Files.readString(datasetPath);
                return TestDataset.fromJson(json);
            } catch (IOException e) {
                throw new RuntimeException("Failed to load dataset: " + name, e);
            }
        });
    }

    /**
     * Get all test images.
     */
    public List<TestImage> getAllImages() throws IOException {
        Path imagesDir = testDataDir.resolve("images");
        if (!Files.exists(imagesDir)) {
            return List.of();
        }
        
        List<TestImage> images = new ArrayList<>();
        try (var stream = Files.list(imagesDir)) {
            stream.filter(Files::isRegularFile)
                  .filter(p -> p.toString().endsWith(".jpg") || p.toString().endsWith(".png"))
                  .forEach(p -> {
                      try {
                          images.add(loadImage(p.getFileName().toString()));
                      } catch (IOException e) {
                          // Skip problematic files
                      }
                  });
        }
        return images;
    }

    /**
     * Create test image batch.
     */
    public List<TestImage> createImageBatch(int size) throws IOException {
        List<TestImage> batch = new ArrayList<>();
        List<TestImage> allImages = getAllImages();
        
        for (int i = 0; i < size && !allImages.isEmpty(); i++) {
            batch.add(allImages.get(i % allImages.size()));
        }
        
        return batch;
    }

    /**
     * Validate test data completeness.
     */
    public TestDataValidationResult validate() {
        TestDataValidationResult result = new TestDataValidationResult();
        
        // Check images directory
        Path imagesDir = testDataDir.resolve("images");
        if (Files.exists(imagesDir)) {
            try {
                long imageCount = Files.list(imagesDir)
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".jpg") || p.toString().endsWith(".png"))
                    .count();
                result.imagesFound = imageCount;
                result.imagesValid = imageCount >= 5; // Minimum 5 test images
            } catch (IOException e) {
                result.errors.add("Failed to list images: " + e.getMessage());
            }
        } else {
            result.errors.add("Images directory not found");
        }
        
        // Check documents directory
        Path docsDir = testDataDir.resolve("documents");
        if (Files.exists(docsDir)) {
            try {
                long docCount = Files.list(docsDir)
                    .filter(Files::isRegularFile)
                    .count();
                result.documentsFound = docCount;
                result.documentsValid = docCount >= 1;
            } catch (IOException e) {
                result.errors.add("Failed to list documents: " + e.getMessage());
            }
        } else {
            result.errors.add("Documents directory not found");
        }
        
        // Check datasets directory
        Path datasetsDir = testDataDir.resolve("datasets");
        if (Files.exists(datasetsDir)) {
            try {
                long datasetCount = Files.list(datasetsDir)
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".json"))
                    .count();
                result.datasetsFound = datasetCount;
                result.datasetsValid = datasetCount >= 3; // Minimum 3 datasets
            } catch (IOException e) {
                result.errors.add("Failed to list datasets: " + e.getMessage());
            }
        } else {
            result.errors.add("Datasets directory not found");
        }
        
        result.isValid = result.imagesValid && result.documentsValid && result.datasetsValid;
        
        return result;
    }

    /**
     * Clear caches.
     */
    public void clearCache() {
        imageCache.clear();
        datasetCache.clear();
    }

    private String detectMimeType(String filename) {
        if (filename.endsWith(".jpg") || filename.endsWith(".jpeg")) {
            return "image/jpeg";
        } else if (filename.endsWith(".png")) {
            return "image/png";
        } else if (filename.endsWith(".webp")) {
            return "image/webp";
        } else if (filename.endsWith(".gif")) {
            return "image/gif";
        }
        return "application/octet-stream";
    }

    private String detectDocType(String filename) {
        if (filename.endsWith(".txt")) {
            return "text/plain";
        } else if (filename.endsWith(".pdf")) {
            return "application/pdf";
        } else if (filename.endsWith(".docx")) {
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
        } else if (filename.endsWith(".html")) {
            return "text/html";
        }
        return "application/octet-stream";
    }

    /**
     * Test image holder.
     */
    public static class TestImage {
        public final String filename;
        public final byte[] bytes;
        public final String mimeType;

        public TestImage(String filename, byte[] bytes, String mimeType) {
            this.filename = filename;
            this.bytes = bytes;
            this.mimeType = mimeType;
        }

        public String getBase64() {
            return Base64.getEncoder().encodeToString(bytes);
        }
    }

    /**
     * Test document holder.
     */
    public static class TestDocument {
        public final String filename;
        public final String content;
        public final String docType;

        public TestDocument(String filename, String content, String docType) {
            this.filename = filename;
            this.content = content;
            this.docType = docType;
        }

        public byte[] getBytes() {
            return content.getBytes();
        }
    }

    /**
     * Test dataset holder.
     */
    public static class TestDataset {
        public final String name;
        public final String version;
        public final String description;
        public final Map<String, Object> data;

        public TestDataset(String name, String version, String description, Map<String, Object> data) {
            this.name = name;
            this.version = version;
            this.description = description;
            this.data = data;
        }

        public static TestDataset fromJson(String json) {
            // Simple JSON parsing - in production use Jackson
            try {
                var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                var node = mapper.readTree(json);
                
                String name = node.path("dataset").asText("");
                String version = node.path("version").asText("");
                String description = node.path("description").asText("");
                
                Map<String, Object> data = mapper.convertValue(node, Map.class);
                
                return new TestDataset(name, version, description, data);
            } catch (Exception e) {
                throw new RuntimeException("Failed to parse dataset JSON", e);
            }
        }
    }

    /**
     * Validation result.
     */
    public static class TestDataValidationResult {
        public boolean isValid;
        public long imagesFound;
        public boolean imagesValid;
        public long documentsFound;
        public boolean documentsValid;
        public long datasetsFound;
        public boolean datasetsValid;
        public List<String> errors = new ArrayList<>();

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Test Data Validation Result:\n");
            sb.append("  Valid: ").append(isValid).append("\n");
            sb.append("  Images: ").append(imagesFound).append(" (").append(imagesValid ? "✓" : "✗").append(")\n");
            sb.append("  Documents: ").append(documentsFound).append(" (").append(documentsValid ? "✓" : "✗").append(")\n");
            sb.append("  Datasets: ").append(datasetsFound).append(" (").append(datasetsValid ? "✓" : "✗").append("\n");
            
            if (!errors.isEmpty()) {
                sb.append("  Errors:\n");
                for (String error : errors) {
                    sb.append("    - ").append(error).append("\n");
                }
            }
            
            return sb.toString();
        }
    }
}
