/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.ml.nn.backend;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Global registry for neural network backend implementations.
 *
 * @author Gollek Team
 * @since 0.2.0
 */
public final class NNBackendRegistry {

    private static final Logger LOG = LoggerFactory.getLogger(NNBackendRegistry.class);

    private static final Map<String, NNBackendProvider> BACKENDS = new ConcurrentHashMap<>();
    private static volatile NNBackendProvider DEFAULT_BACKEND;
    private static volatile boolean initialized = false;

    private NNBackendRegistry() {}

    public static synchronized void register(NNBackendProvider provider) {
        if (provider == null) {
            throw new NullPointerException("Backend provider cannot be null");
        }
        BACKENDS.put(provider.getBackendId(), provider);
        LOG.info("Registered NN backend: {} (priority: {})", provider.getBackendId(), provider.getPriority());
        DEFAULT_BACKEND = null;
    }

    public static NNBackendProvider get(String backendId) {
        NNBackendProvider provider = BACKENDS.get(backendId);
        if (provider == null) {
            throw new IllegalArgumentException("NN backend not found: " + backendId);
        }
        return provider;
    }

    public static NNBackendProvider getDefault() {
        if (DEFAULT_BACKEND != null) {
            return DEFAULT_BACKEND;
        }

        synchronized (NNBackendRegistry.class) {
            if (DEFAULT_BACKEND != null) {
                return DEFAULT_BACKEND;
            }

            if (BACKENDS.isEmpty()) {
                // Try to trigger CPUNNBackend registration if not already done
                try {
                    Class.forName("tech.kayys.gollek.ml.nn.backend.CPUNNBackend");
                } catch (ClassNotFoundException e) {
                    LOG.error("Failed to auto-load CPUNNBackend", e);
                }
            }

            if (BACKENDS.isEmpty()) {
                throw new IllegalStateException("No NN backends registered.");
            }

            NNBackendProvider best = BACKENDS.values()
                    .stream()
                    .filter(NNBackendProvider::isAvailable)
                    .min(Comparator.comparingInt(NNBackendProvider::getPriority))
                    .orElseThrow(() -> new IllegalStateException("No available NN backends."));

            DEFAULT_BACKEND = best;
            return DEFAULT_BACKEND;
        }
    }

    public static void clear() {
        BACKENDS.clear();
        DEFAULT_BACKEND = null;
        initialized = false;
    }
}
