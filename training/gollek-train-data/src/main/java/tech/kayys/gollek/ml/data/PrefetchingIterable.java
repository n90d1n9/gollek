package tech.kayys.gollek.ml.data;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Bounded asynchronous iterable wrapper for overlapping ML data loading with trainer compute.
 */
public final class PrefetchingIterable<T> implements Iterable<T>, AutoCloseable {
    private static final AtomicLong WORKER_IDS = new AtomicLong();
    private static final Signal<?> END = new Signal<>(SignalKind.END, null, null);

    private final Iterable<? extends T> source;
    private final int bufferSize;
    private final Set<PrefetchingIterator<T>> activeIterators = ConcurrentHashMap.newKeySet();
    private final AtomicBoolean closed = new AtomicBoolean();

    public PrefetchingIterable(Iterable<? extends T> source, int bufferSize) {
        this.source = Objects.requireNonNull(source, "source must not be null");
        DataLoaderBatchSizes.requirePositive(bufferSize);
        this.bufferSize = bufferSize;
    }

    public PrefetchingIterable(Iterable<? extends T> source) {
        this(source, DataLoaderPrefetchPlan.DEFAULT_BUFFER_SIZE);
    }

    public static <T> PrefetchingIterable<T> of(Iterable<? extends T> source, int bufferSize) {
        return new PrefetchingIterable<>(source, bufferSize);
    }

    public static <T> PrefetchingIterable<T> of(Iterable<? extends T> source) {
        return new PrefetchingIterable<>(source);
    }

    public int bufferSize() {
        return bufferSize;
    }

    public Iterable<? extends T> source() {
        return source;
    }

    public DataLoaderPrefetchPlan plan() {
        return DataLoaderPrefetchPlan.enabled(bufferSize);
    }

    public static DataLoaderPrefetchPlan disabledPlan() {
        return DataLoaderPrefetchPlan.disabled();
    }

    @Override
    public CloseableIterator<T> iterator() {
        if (closed.get()) {
            throw new IllegalStateException("prefetching iterable is already closed");
        }
        PrefetchingIterator<T> iterator = new PrefetchingIterator<>(source.iterator(), bufferSize);
        activeIterators.add(iterator);
        iterator.onClose(() -> activeIterators.remove(iterator));
        iterator.start();
        return iterator;
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        for (PrefetchingIterator<T> iterator : Set.copyOf(activeIterators)) {
            iterator.close();
        }
        activeIterators.clear();
    }

    public interface CloseableIterator<T> extends Iterator<T>, AutoCloseable {
        @Override
        void close();
    }

    private static final class PrefetchingIterator<T> implements CloseableIterator<T> {
        private final Iterator<? extends T> source;
        private final BlockingQueue<Signal<T>> queue;
        private final AtomicBoolean closed = new AtomicBoolean();
        private Runnable onClose = () -> {};
        private Thread worker;
        private Signal<T> nextSignal;
        private boolean finished;

        private PrefetchingIterator(Iterator<? extends T> source, int bufferSize) {
            this.source = Objects.requireNonNull(source, "source iterator must not be null");
            this.queue = new ArrayBlockingQueue<>(bufferSize);
        }

        private void onClose(Runnable onClose) {
            this.onClose = Objects.requireNonNull(onClose, "onClose must not be null");
        }

        private void start() {
            worker = new Thread(this::runWorker, "gollek-ml-data-prefetch-" + WORKER_IDS.incrementAndGet());
            worker.setDaemon(true);
            worker.start();
        }

        @Override
        public boolean hasNext() {
            if (nextSignal != null) {
                return true;
            }
            if (finished || closed.get()) {
                return false;
            }
            nextSignal = takeSignal();
            if (nextSignal == null || nextSignal.kind == SignalKind.END) {
                finished = true;
                close();
                nextSignal = null;
                return false;
            }
            if (nextSignal.kind == SignalKind.ERROR) {
                finished = true;
                close();
                throw propagate(nextSignal.error);
            }
            return true;
        }

        @Override
        public T next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            Signal<T> signal = nextSignal;
            nextSignal = null;
            return signal.value;
        }

        @Override
        public void close() {
            if (!closed.compareAndSet(false, true)) {
                return;
            }
            Thread currentWorker = worker;
            if (currentWorker != null) {
                currentWorker.interrupt();
            }
            queue.clear();
            onClose.run();
        }

        private void runWorker() {
            try {
                while (!closed.get() && source.hasNext()) {
                    enqueue(Signal.value(source.next()));
                }
            } catch (Throwable error) {
                enqueue(Signal.error(error));
            } finally {
                enqueue(endSignal());
            }
        }

        private void enqueue(Signal<T> signal) {
            while (!closed.get()) {
                try {
                    if (queue.offer(signal, 100, TimeUnit.MILLISECONDS)) {
                        return;
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }

        private Signal<T> takeSignal() {
            while (!closed.get()) {
                try {
                    Signal<T> signal = queue.poll(100, TimeUnit.MILLISECONDS);
                    if (signal != null) {
                        return signal;
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    closed.set(true);
                    throw new IllegalStateException("interrupted while waiting for prefetched data", e);
                }
            }
            return null;
        }

        private static RuntimeException propagate(Throwable error) {
            if (error instanceof RuntimeException runtimeException) {
                return runtimeException;
            }
            if (error instanceof Error fatal) {
                throw fatal;
            }
            return new IllegalStateException("prefetch worker failed", error);
        }
    }

    private enum SignalKind {
        VALUE,
        ERROR,
        END
    }

    private static final class Signal<T> {
        private final SignalKind kind;
        private final T value;
        private final Throwable error;

        private Signal(SignalKind kind, T value, Throwable error) {
            this.kind = kind;
            this.value = value;
            this.error = error;
        }

        private static <T> Signal<T> value(T value) {
            return new Signal<>(SignalKind.VALUE, value, null);
        }

        private static <T> Signal<T> error(Throwable error) {
            return new Signal<>(SignalKind.ERROR, null, error);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> Signal<T> endSignal() {
        return (Signal<T>) END;
    }
}
