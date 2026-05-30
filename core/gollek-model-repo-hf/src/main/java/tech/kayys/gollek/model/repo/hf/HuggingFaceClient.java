package tech.kayys.gollek.model.repo.hf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import com.fasterxml.jackson.databind.ObjectMapper;
import tech.kayys.gollek.model.download.DownloadProgressListener;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;

/**
 * Client for HuggingFace API
 */
@ApplicationScoped
public class HuggingFaceClient {

    private static final Logger LOG = Logger.getLogger(HuggingFaceClient.class);
    private static final int DOWNLOAD_MAX_ATTEMPTS = 5;
    private static final long DOWNLOAD_INITIAL_RETRY_DELAY_MS = 1_000L;

    @Inject
    HuggingFaceConfig config;

    @Inject
    ObjectMapper objectMapper;

    private final HttpClient httpClient;

    public HuggingFaceClient() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    @jakarta.annotation.PostConstruct
    void init() {
        objectMapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());
    }

    /**
     * Get model information
     */
    public HuggingFaceModelInfo getModelInfo(String modelId) throws IOException, InterruptedException {
        return getModelInfo(modelId, config.revision());
    }

    public HuggingFaceModelInfo getModelInfo(String modelId, String revision) throws IOException, InterruptedException {
        String effectiveRevision = revision != null ? revision : config.revision();
        String url;
        if (effectiveRevision != null && !effectiveRevision.trim().isEmpty() && !effectiveRevision.equalsIgnoreCase("main") && !effectiveRevision.equalsIgnoreCase("master")) {
            url = config.baseUrl() + "/api/models/" + modelId + "/revision/" + java.net.URLEncoder.encode(effectiveRevision.trim(), java.nio.charset.StandardCharsets.UTF_8) + "?siblings=true";
        } else {
            url = config.baseUrl() + "/api/models/" + modelId + "?siblings=true";
        }

        // Fetch model info with siblings (files)
        return fetchModelInfo(url);
    }

    /**
     * List files in a model repository
     */
    public List<String> listFiles(String modelId) throws IOException, InterruptedException {
        return listFiles(modelId, config.revision());
    }

    public List<String> listFiles(String modelId, String revision) throws IOException, InterruptedException {
        HuggingFaceModelInfo info = getModelInfo(modelId, revision);
        if (info.getFiles() == null) {
            return List.of();
        }
        return info.getFiles().stream()
                .map(HuggingFaceModelInfo.ModelFile::getFilename)
                .toList();
    }

    private HuggingFaceModelInfo fetchModelInfo(String url) throws IOException, InterruptedException {
        LOG.infof("Fetching model info from: %s", url);

        HttpResponse<String> response = httpClient.send(
                buildGetRequest(url, true),
                HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 401 && hasUsableToken()) {
            response = httpClient.send(
                    buildGetRequest(url, true),
                    HttpResponse.BodyHandlers.ofString());
        }

        if (response.statusCode() != 200) {
            String body = response.body();
            if (body == null) {
                body = "";
            }
            body = body.replaceAll("\\s+", " ").trim();
            if (body.length() > 512) {
                body = body.substring(0, 512) + "...";
            }
            throw new IOException(String.format(
                    "Failed to fetch model info: %d - %s",
                    response.statusCode(),
                    body));
        }

        String json = response.body();
        try {
            return parseModelInfo(json);
        } catch (IOException e) {
            LOG.errorf("Failed to parse HuggingFace model info. Body: %s", json);
            throw e;
        }
    }

    /**
     * Download a specific file from a model
     */
    public void downloadFile(
            String modelId,
            String filename,
            java.nio.file.Path targetPath,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {
        downloadFile(modelId, filename, config.revision(), targetPath, progressListener);
    }

    public void downloadFile(
            String modelId,
            String filename,
            String revision,
            java.nio.file.Path targetPath,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {
        String effectiveRevision = revision != null ? revision : config.revision();
        String url = String.format(
                "%s/%s/resolve/%s/%s",
                config.baseUrl(),
                modelId,
                effectiveRevision,
                filename);

        LOG.infof("Downloading: %s from %s", filename, modelId);

        Files.createDirectories(targetPath.getParent());
        java.nio.file.Path partialPath = targetPath.resolveSibling(targetPath.getFileName() + ".part");
        IOException lastFailure = null;
        long retryDelayMs = DOWNLOAD_INITIAL_RETRY_DELAY_MS;

        for (int attempt = 1; attempt <= DOWNLOAD_MAX_ATTEMPTS; attempt++) {
            long resumeFrom = Files.exists(partialPath) ? Files.size(partialPath) : 0L;
            HttpResponse<InputStream> response = null;
            try {
                response = openDownloadResponse(url, resumeFrom);
                int status = response.statusCode();

                if (status == 416 && resumeFrom > 0L) {
                    long remoteTotal = parseUnsatisfiedRangeTotal(response);
                    closeQuietly(response.body());
                    if (remoteTotal > 0L && resumeFrom == remoteTotal) {
                        Files.move(partialPath, targetPath, StandardCopyOption.REPLACE_EXISTING);
                        notifyDownloadComplete(progressListener, remoteTotal);
                        LOG.infof("Downloaded: %s (%d bytes)", filename, Files.size(targetPath));
                        return;
                    }
                    LOG.warnf("Discarding unusable partial download for %s (%d/%d bytes).",
                            filename, resumeFrom, remoteTotal);
                    Files.deleteIfExists(partialPath);
                    continue;
                }

                if (status == 200 && resumeFrom > 0L) {
                    LOG.warnf("Server ignored range resume for %s; restarting this file.", filename);
                    closeQuietly(response.body());
                    Files.deleteIfExists(partialPath);
                    resumeFrom = 0L;
                    response = openDownloadResponse(url, 0L);
                    status = response.statusCode();
                }

                if (status != 200 && status != 206) {
                    throw downloadStatusException(response);
                }

                long totalBytes = resolveTotalBytes(response, resumeFrom);
                if (resumeFrom == 0L && Files.exists(targetPath) && totalBytes > 0L) {
                    long existingSize = Files.size(targetPath);
                    if (existingSize == totalBytes) {
                        closeQuietly(response.body());
                        LOG.infof("File already exists and matches size (%d bytes), skipping download.", existingSize);
                        notifyDownloadComplete(progressListener, totalBytes);
                        return;
                    }
                    if (existingSize > 0L && existingSize < totalBytes) {
                        closeQuietly(response.body());
                        LOG.infof("Resuming existing partial file %s (%d/%d bytes).",
                                filename, existingSize, totalBytes);
                        Files.move(targetPath, partialPath, StandardCopyOption.REPLACE_EXISTING);
                        continue;
                    }
                    if (existingSize > totalBytes) {
                        closeQuietly(response.body());
                        LOG.warnf("Existing file %s is larger than remote (%d > %d); restarting.",
                                filename, existingSize, totalBytes);
                        Files.deleteIfExists(targetPath);
                        continue;
                    }
                }

                try (InputStream is = response.body()) {
                    downloadWithProgress(is, partialPath, totalBytes, resumeFrom, progressListener);
                }

                long downloadedSize = Files.size(partialPath);
                if (totalBytes > 0L && downloadedSize != totalBytes) {
                    throw new IOException("Incomplete download for " + filename
                            + ": expected " + totalBytes + " bytes, received " + downloadedSize);
                }

                Files.move(partialPath, targetPath, StandardCopyOption.REPLACE_EXISTING);
                notifyDownloadComplete(progressListener, downloadedSize);
                LOG.infof("Downloaded: %s (%d bytes)", filename, Files.size(targetPath));
                return;
            } catch (IOException e) {
                closeQuietly(response == null ? null : response.body());
                lastFailure = e;
                if (attempt >= DOWNLOAD_MAX_ATTEMPTS) {
                    if (progressListener != null) {
                        progressListener.onError(e);
                    }
                    throw e;
                }
                LOG.warnf("Download attempt %d/%d failed for %s: %s. Retrying with resume.",
                        attempt, DOWNLOAD_MAX_ATTEMPTS, filename, e.getMessage());
                Thread.sleep(retryDelayMs);
                retryDelayMs = Math.min(retryDelayMs * 2L, 10_000L);
            }
        }

        if (lastFailure != null) {
            throw lastFailure;
        }
    }

    /**
     * Download entire repository (essential files for conversion)
     */
    public void downloadRepository(String modelId, java.nio.file.Path targetDir,
            DownloadProgressListener progressListener)
            throws IOException, InterruptedException {
        downloadRepository(modelId, config.revision(), targetDir, progressListener);
    }

    public void downloadRepository(String modelId, String revision, java.nio.file.Path targetDir,
            DownloadProgressListener progressListener)
            throws IOException, InterruptedException {
        List<String> files = listFiles(modelId, revision);

        // Filter for essential files if you want to optimize,
        // but for conversion we might need everything except maybe other safetensors if
        // checking strictly?
        // For simplicity, download everything or at least:
        // - config.json, tokenizer.json, *.safetensors, *.bin, tokenizer_config.json

        for (String file : files) {
            // Skip .git attributes or large hidden files if any
            if (file.startsWith("."))
                continue;

            downloadFile(modelId, file, revision, targetDir.resolve(file), progressListener);
        }
    }

    private void downloadWithProgress(
            InputStream inputStream,
            java.nio.file.Path targetPath,
            long totalBytes,
            long alreadyDownloadedBytes,
            DownloadProgressListener progressListener) throws IOException, InterruptedException {

        Files.createDirectories(targetPath.getParent());

        byte[] buffer = new byte[1024 * 1024];
        long downloadedBytes = alreadyDownloadedBytes;
        int bytesRead;

        // Register shutdown hook to interrupt download on Ctrl+C
        Thread currentThread = Thread.currentThread();
        Thread shutdownHook = new Thread(() -> {
            try {
                // Interrupt the download thread to stop the loop/IO
                currentThread.interrupt();
                // Force immediate exit with standard Ctrl+C exit code
                System.exit(130);
            } catch (Exception e) {
                // Best effort
            }
        });
        Runtime.getRuntime().addShutdownHook(shutdownHook);

        if (progressListener != null && totalBytes > 0L && alreadyDownloadedBytes > 0L) {
            progressListener.onProgress(alreadyDownloadedBytes, totalBytes,
                    Math.min(1.0, (double) alreadyDownloadedBytes / totalBytes));
        }

        var options = alreadyDownloadedBytes > 0L
                ? new java.nio.file.StandardOpenOption[] {
                        java.nio.file.StandardOpenOption.CREATE,
                        java.nio.file.StandardOpenOption.APPEND
                }
                : new java.nio.file.StandardOpenOption[] {
                        java.nio.file.StandardOpenOption.CREATE,
                        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
                        java.nio.file.StandardOpenOption.WRITE
                };

        try (var outputStream = Files.newOutputStream(targetPath, options)) {
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                if (Thread.currentThread().isInterrupted()) {
                    throw new InterruptedException("Download interrupted");
                }
                outputStream.write(buffer, 0, bytesRead);
                downloadedBytes += bytesRead;

                if (progressListener != null && totalBytes > 0) {
                    double progress = (double) downloadedBytes / totalBytes;
                    progressListener.onProgress(downloadedBytes, totalBytes, progress);
                }
            }
        } finally {
            // Remove hook if finished normally
            try {
                Runtime.getRuntime().removeShutdownHook(shutdownHook);
            } catch (IllegalStateException e) {
                // Ignore if shutdown is in progress
            }
        }
    }

    private HttpResponse<InputStream> openDownloadResponse(String url, long resumeFrom)
            throws IOException, InterruptedException {
        HttpResponse<InputStream> response = httpClient.send(
                buildGetRequest(url, true, 600, resumeFrom),
                HttpResponse.BodyHandlers.ofInputStream());

        if (response.statusCode() == 401 && hasUsableToken()) {
            closeQuietly(response.body());
            response = httpClient.send(
                    buildGetRequest(url, true, 600, resumeFrom),
                    HttpResponse.BodyHandlers.ofInputStream());
        }
        return response;
    }

    private IOException downloadStatusException(HttpResponse<InputStream> response) {
        String detail = "";
        try (InputStream errorBody = response.body()) {
            byte[] bytes = errorBody.readNBytes(512);
            if (bytes.length > 0) {
                detail = " - " + new String(bytes, java.nio.charset.StandardCharsets.UTF_8)
                        .replaceAll("\\s+", " ")
                        .trim();
            }
        } catch (Exception ignored) {
            // best effort
        }
        return new IOException(String.format(
                "Failed to download file: %d%s",
                response.statusCode(),
                detail));
    }

    private long resolveTotalBytes(HttpResponse<InputStream> response, long resumeFrom) {
        long rangeTotal = parseContentRangeTotal(response);
        if (rangeTotal > 0L) {
            return rangeTotal;
        }
        long contentLength = response.headers()
                .firstValueAsLong("Content-Length")
                .orElse(-1L);
        if (contentLength <= 0L) {
            return -1L;
        }
        return response.statusCode() == 206 ? resumeFrom + contentLength : contentLength;
    }

    private long parseContentRangeTotal(HttpResponse<?> response) {
        return response.headers()
                .firstValue("Content-Range")
                .map(this::parseContentRangeTotal)
                .orElse(-1L);
    }

    private long parseUnsatisfiedRangeTotal(HttpResponse<?> response) {
        return parseContentRangeTotal(response);
    }

    private long parseContentRangeTotal(String contentRange) {
        if (contentRange == null || contentRange.isBlank()) {
            return -1L;
        }
        int slash = contentRange.lastIndexOf('/');
        if (slash < 0 || slash + 1 >= contentRange.length()) {
            return -1L;
        }
        String total = contentRange.substring(slash + 1).trim();
        if (total.equals("*")) {
            return -1L;
        }
        try {
            return Long.parseLong(total);
        } catch (NumberFormatException ignored) {
            return -1L;
        }
    }

    private void notifyDownloadComplete(DownloadProgressListener progressListener, long totalBytes) {
        if (progressListener == null) {
            return;
        }
        if (totalBytes > 0L) {
            progressListener.onProgress(totalBytes, totalBytes, 1.0);
        }
        progressListener.onComplete(totalBytes);
    }

    private void closeQuietly(InputStream stream) {
        if (stream == null) {
            return;
        }
        try {
            stream.close();
        } catch (IOException ignored) {
        }
    }

    private HuggingFaceModelInfo parseModelInfo(String json) throws IOException {
        try {
            return objectMapper.readValue(json, HuggingFaceModelInfo.class);
        } catch (Exception e) {
            throw new IOException("Failed to parse model info: " + e.getMessage(), e);
        }
    }

    private HttpRequest buildGetRequest(String url, boolean withAuth) {
        return buildGetRequest(url, withAuth, config.timeoutSeconds());
    }

    private HttpRequest buildGetRequest(String url, boolean withAuth, int timeoutSeconds) {
        return buildGetRequest(url, withAuth, timeoutSeconds, 0L);
    }

    private HttpRequest buildGetRequest(String url, boolean withAuth, int timeoutSeconds, long rangeStart) {
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(timeoutSeconds))
                .header("User-Agent", config.userAgent())
                .GET();
        if (rangeStart > 0L) {
            requestBuilder.header("Range", "bytes=" + rangeStart + "-");
        }
        if (withAuth) {
            configuredToken().ifPresent(token -> requestBuilder.header("Authorization", "Bearer " + token));
        }
        return requestBuilder.build();
    }

    private java.util.Optional<String> configuredToken() {
        java.util.Optional<String> configured = config.token()
                .map(String::trim)
                .filter(this::isUsableToken);
        if (configured.isPresent()) {
            return configured;
        }

        // CLI usually maps env vars into config/system props, but keep a direct
        // fallback for packaged/runtime variants where that mapping is skipped.
        String[] fallbackKeys = new String[] {
                "wayang.inference.repository.huggingface.token",
                "WAYANG_INFERENCE_REPOSITORY_HUGGINGFACE_TOKEN",
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN"
        };
        for (String key : fallbackKeys) {
            String value = key.contains(".") ? System.getProperty(key) : System.getenv(key);
            if (isUsableToken(value)) {
                return java.util.Optional.of(value.trim());
            }
        }
        return java.util.Optional.empty();
    }

    private boolean hasUsableToken() {
        return configuredToken().isPresent();
    }

    private boolean isUsableToken(String token) {
        if (token == null || token.isBlank()) {
            return false;
        }
        String normalized = token.trim().toLowerCase();
        if (normalized.equals("dummy")
                || normalized.equals("null")
                || normalized.equals("none")
                || normalized.equals("changeme")
                || normalized.equals("your_token_here")) {
            return false;
        }
        return !token.contains("${");
    }
}
