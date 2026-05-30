package tech.kayys.gollek.train.data;

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
 * Bounded asynchronous prefetch wrapper for data-loading pipelines.
 *
 * <p>Each iterator owns one daemon worker that pulls from the source iterator and
 * buffers up to {@code bufferSize} elements. Fully consuming the iterator closes
 * the worker automatically. For early exits, prefer try-with-resources around
 * the iterable so the worker can be interrupted promptly.</p>
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
        if (bufferSize <= 0) {
            throw new IllegalArgumentException("bufferSize must be positive, got: " + bufferSize);
        }
        this.bufferSize = bufferSize;
    }

    public static <T> PrefetchingIterable<T> of(Iterable<? extends T> source, int bufferSize) {
        return new PrefetchingIterable<>(source, bufferSize);
    }

    public int bufferSize() {
        return bufferSize;
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
            worker = new Thread(this::runWorker, "gollek-data-prefetch-" + WORKER_IDS.incrementAndGet());
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
