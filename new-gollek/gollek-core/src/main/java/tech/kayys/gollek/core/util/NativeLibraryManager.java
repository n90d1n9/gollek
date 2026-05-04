package tech.kayys.gollek.core.util;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Centralized native library management for Gollek inference engines.
 * <p>
 * Provides a standard location for native libraries at {@code ~/.gollek/libs/}
 * with automatic extraction from JAR resources and platform-aware loading.
 * </p>
 * <p>
 * Supported libraries:
 * <ul>
 * <li>llama.cpp (GGUF runner)</li>
 * <li>ONNX Runtime (ONNX runner)</li>
 * <li>LibTorch (PyTorch runner)</li>
 * <li>TensorFlow Lite (TFLite runner)</li>
 * </ul>
 * </p>
 *
 * <h2>Usage Example</h2>
 * 
 * <pre>{@code
 * // Load llama.cpp library
 * NativeLibraryManager manager = new NativeLibraryManager("llama");
 * Path libraryPath = manager.getOrCreateLibraryPath();
 * manager.loadLibrary();
 *
 * // Or with explicit source
 * Path sourceLib = Path.of("/build/libllama.dylib");
 * manager.copyToStandardLocation(sourceLib);
 * manager.loadLibrary();
 * }</pre>
 *
 * <h2>Environment Variables</h2>
 * <ul>
 * <li>{@code GOLLEK_NATIVE_LIB_DIR} - Override default
 * {@code ~/.gollek/libs/}</li>
 * <li>{@code GOLLEK_LLAMA_LIB_DIR} - Specific to llama.cpp</li>
 * <li>{@code GOLLEK_ONNX_LIB_DIR} - Specific to ONNX Runtime</li>
 * </ul>
 */
public class NativeLibraryManager {

    private static final Logger log = Logger.getLogger(NativeLibraryManager.class);
    private static final String DEFAULT_BASE_DIR = Path.of(System.getProperty("user.home"), ".gollek", "libs")
            .toString();

    private final String libraryName;
    private final String libraryBaseName;
    private final Path baseDir;
    private Path installedLibraryPath;

    /**
     * Create a native library manager for the specified library.
     *
     * @param libraryName the library name (e.g., "llama", "onnxruntime", "torch")
     */
    public NativeLibraryManager(String libraryName) {
        this.libraryName = libraryName;
        this.libraryBaseName = normalizeLibraryName(libraryName);
        this.baseDir = resolveBaseDirectory();
    }

    /**
     * Get the base directory for native libraries.
     *
     * @return the base directory path
     */
    public static Path getBaseDirectory() {
        return Path.of(resolveBaseDirectoryFromEnv());
    }

    /**
     * Resolve the base directory for native libraries.
     * Uses environment variable override or default location.
     *
     * @return the base directory path
     */
    private static String resolveBaseDirectoryFromEnv() {
        String gollekNativeDir = System.getenv("GOLLEK_NATIVE_LIB_DIR");
        if (gollekNativeDir != null && !gollekNativeDir.isBlank()) {
            log.debugf("Using GOLLEK_NATIVE_LIB_DIR: %s", gollekNativeDir);
            return gollekNativeDir.trim();
        }
        log.debugf("Using default native lib directory: %s", DEFAULT_BASE_DIR);
        return DEFAULT_BASE_DIR;
    }

    private Path resolveBaseDirectory() {
        String gollekNativeDir = System.getenv("GOLLEK_NATIVE_LIB_DIR");
        if (gollekNativeDir != null && !gollekNativeDir.isBlank()) {
            return Path.of(gollekNativeDir.trim());
        }
        return Path.of(DEFAULT_BASE_DIR);
    }

    /**
     * Get or create the standard library directory for this library.
     *
     * @return the library directory path
     */
    public Path getLibraryDirectory() {
        Path libDir = baseDir.resolve(libraryName);
        if (!Files.exists(libDir)) {
            try {
                Files.createDirectories(libDir);
                log.debugf("Created native library directory: %s", libDir);
            } catch (IOException e) {
                log.errorf("Failed to create native library directory %s: %s", libDir, e.getMessage());
            }
        }
        return libDir;
    }

    /**
     * Get the expected library file path.
     *
     * @return the expected library file path
     */
    public Path getLibraryFilePath() {
        String libFileName = getLibraryFileName();
        return getLibraryDirectory().resolve(libFileName);
    }

    /**
     * Get the platform-specific library file name.
     *
     * @return the library file name (e.g., "libllama.dylib" on macOS)
     */
    public String getLibraryFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin")) {
            return "lib" + libraryBaseName + ".dylib";
        } else if (os.contains("win")) {
            return libraryBaseName + ".dll";
        } else {
            return "lib" + libraryBaseName + ".so";
        }
    }

    /**
     * Get the platform-specific library file extension.
     *
     * @return the library extension (e.g., ".dylib", ".so", ".dll")
     */
    public static String getNativeExtension() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac") || os.contains("darwin")) {
            return ".dylib";
        } else if (os.contains("win")) {
            return ".dll";
        } else {
            return ".so";
        }
    }

    /**
     * Copy a library file to the standard location.
     *
     * @param sourcePath the source library path
     * @return the destination path
     * @throws IOException if the copy fails
     */
    public Path copyToStandardLocation(Path sourcePath) throws IOException {
        if (!Files.exists(sourcePath)) {
            throw new IOException("Source library does not exist: " + sourcePath);
        }

        Path destDir = getLibraryDirectory();
        Path destPath = destDir.resolve(sourcePath.getFileName());

        Files.createDirectories(destDir);
        Files.copy(sourcePath, destPath, StandardCopyOption.REPLACE_EXISTING);

        // Set executable on Unix-like systems
        if (!isWindows()) {
            destPath.toFile().setExecutable(true);
        }

        log.infof("Copied native library %s to standard location: %s", sourcePath.getFileName(), destPath);
        this.installedLibraryPath = destPath;
        return destPath;
    }

    /**
     * Copy multiple library files (main + dependencies) to the standard location.
     *
     * @param sourceDir    the source directory containing libraries
     * @param libraryNames names of libraries to copy
     * @return list of destination paths
     * @throws IOException if copy operations fail
     */
    public List<Path> copyLibrariesToStandardLocation(Path sourceDir, List<String> libraryNames) throws IOException {
        List<Path> copiedPaths = new ArrayList<>();
        Path destDir = getLibraryDirectory();
        Files.createDirectories(destDir);

        for (String libName : libraryNames) {
            String fileName = normalizeLibraryName(libName);
            String fullFileName = "lib" + fileName + getNativeExtension();
            Path sourcePath = sourceDir.resolve(fullFileName);

            if (Files.exists(sourcePath)) {
                Path destPath = destDir.resolve(fullFileName);
                Files.copy(sourcePath, destPath, StandardCopyOption.REPLACE_EXISTING);

                if (!isWindows()) {
                    destPath.toFile().setExecutable(true);
                }

                copiedPaths.add(destPath);
                log.debugf("Copied %s to %s", fullFileName, destPath);
            } else {
                log.debugf("Library not found: %s", sourcePath);
            }
        }

        if (!copiedPaths.isEmpty()) {
            log.infof("Copied %d native libraries to %s", copiedPaths.size(), destDir);
        }

        return copiedPaths;
    }

    /**
     * Load the native library using System.load().
     *
     * @return true if loaded successfully
     */
    public boolean loadLibrary() {
        Path libPath = getLibraryFilePath();
        if (!Files.exists(libPath)) {
            log.warnf("Native library not found at %s", libPath);
            return false;
        }

        try {
            System.load(libPath.toAbsolutePath().toString());
            log.infof("Loaded native library: %s", libPath);
            return true;
        } catch (UnsatisfiedLinkError e) {
            log.errorf("Failed to load native library %s: %s", libPath, e.getMessage());
            return false;
        }
    }

    /**
     * Load the native library from a specific path.
     *
     * @param libPath the library path
     * @return true if loaded successfully
     */
    public static boolean loadLibrary(Path libPath) {
        if (!Files.exists(libPath)) {
            log.warnf("Native library not found at %s", libPath);
            return false;
        }

        try {
            System.load(libPath.toAbsolutePath().toString());
            log.infof("Loaded native library: %s", libPath);
            return true;
        } catch (UnsatisfiedLinkError e) {
            log.errorf("Failed to load native library %s: %s", libPath, e.getMessage());
            return false;
        }
    }

    /**
     * Check if the library exists in the standard location.
     *
     * @return true if the library file exists
     */
    public boolean libraryExists() {
        return Files.exists(getLibraryFilePath());
    }

    /**
     * Get the installed library path if loaded.
     *
     * @return the installed library path or empty
     */
    public Optional<Path> getInstalledLibraryPath() {
        return Optional.ofNullable(installedLibraryPath);
    }

    /**
     * Clear macOS quarantine attributes for a library directory.
     *
     * @param libDir the library directory
     */
    public static void clearMacQuarantine(Path libDir) {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (!(os.contains("mac") || os.contains("darwin"))) {
            return;
        }

        Path xattr = Path.of("/usr/bin/xattr");
        if (!Files.isExecutable(xattr)) {
            return;
        }

        try {
            Process process = new ProcessBuilder(
                    xattr.toString(), "-dr", "com.apple.quarantine", libDir.toAbsolutePath().toString())
                    .redirectErrorStream(true)
                    .start();
            int exit = process.waitFor();
            if (exit == 0) {
                log.debugf("Cleared macOS quarantine flags for: %s", libDir);
            } else {
                log.debugf("xattr quarantine cleanup exited with code %d for %s", exit, libDir);
            }
        } catch (Exception e) {
            log.debugf("Unable to clear macOS quarantine for %s: %s", libDir, e.getMessage());
        }
    }

    private static String normalizeLibraryName(String name) {
        // Remove "lib" prefix if present
        if (name.startsWith("lib")) {
            name = name.substring(3);
        }
        // Remove extension if present
        if (name.endsWith(".dylib") || name.endsWith(".so") || name.endsWith(".dll")) {
            name = name.substring(0, name.lastIndexOf('.'));
        }
        return name;
    }

    private static boolean isWindows() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("win");
    }
}
