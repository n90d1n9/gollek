package tech.kayys.gollek.train.diffusion.opd.adapter;

import java.util.Map;
import java.util.Objects;
import tech.kayys.gollek.core.backend.ComputeBackend;
import tech.kayys.gollek.core.memory.CpuBuffer;
import tech.kayys.gollek.core.tensor.DType;
import tech.kayys.gollek.core.tensor.DefaultTensor;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.core.tensor.Shape;
import tech.kayys.gollek.metal.binding.MetalBinding;
import tech.kayys.gollek.core.tensor.Tensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.runner.sd.PNDMScheduler;
import tech.kayys.gollek.safetensor.runner.sd.UNetModel;
import tech.kayys.gollek.train.diffusion.api.DiffusionDenoiser;
import tech.kayys.gollek.train.diffusion.api.DiffusionScheduler;

/**
 * Adapters from the safetensor-native Stable Diffusion runner surfaces to the
 * Java diffusion OPD contracts.
 *
 * <p>This is the bridge point for aligning Gollek's inference-side Stable
 * Diffusion runtime with the Java-first diffusion training stack. The
 * scheduler path is directly adaptable, while the UNet path requires an
 * explicit tensor bridge because the stable-diffusion-native runner uses
 * {@link AccelTensor} and OPD uses core {@link Tensor}.
 */
public final class StableDiffusionRunnerAdapters {

    private static final ComputeBackend MATERIALIZATION_ONLY_BACKEND =
            new MaterializationOnlyBackend("stable-diffusion-bridge-materializer");

    private StableDiffusionRunnerAdapters() {
    }

    public static DiffusionScheduler scheduler(PNDMScheduler scheduler) {
        Objects.requireNonNull(scheduler, "scheduler must not be null");
        return new DiffusionScheduler() {
            @Override
            public Tensor step(Tensor xT, Tensor modelPrediction, int timestepIndex) {
                throw new UnsupportedOperationException(
                        "Stable Diffusion PNDM scheduler step requires an AccelTensor bridge. "
                                + "Use scheduler(PNDMScheduler, TensorAccelBridge) or "
                                + "scheduler(PNDMScheduler, MetalBinding) for executable step alignment.");
            }

            @Override
            public int[] timesteps() {
                return scheduler.timestepsArray();
            }
        };
    }

    public static DiffusionScheduler scheduler(PNDMScheduler scheduler, TensorAccelBridge tensorBridge) {
        Objects.requireNonNull(scheduler, "scheduler must not be null");
        Objects.requireNonNull(tensorBridge, "tensorBridge must not be null");
        return new DiffusionScheduler() {
            @Override
            public Tensor step(Tensor xT, Tensor modelPrediction, int timestepIndex) {
                int[] timesteps = scheduler.timestepsArray();
                if (timestepIndex < 0 || timestepIndex >= timesteps.length) {
                    throw new IllegalArgumentException(
                            "timestepIndex out of range: " + timestepIndex + " for " + timesteps.length + " timesteps");
                }
                try (AccelTensor accelXT = tensorBridge.toAccelTensor(xT);
                        AccelTensor accelPrediction = tensorBridge.toAccelTensor(modelPrediction);
                        AccelTensor stepped = scheduler.step(accelPrediction, timesteps[timestepIndex], accelXT)) {
                    return tensorBridge.toCoreTensor(stepped);
                }
            }

            @Override
            public int[] timesteps() {
                return scheduler.timestepsArray();
            }
        };
    }

    /**
     * Convenience executable scheduler bridge for the maintained Metal binding
     * surface plus materialized core-tensor transport.
     */
    public static DiffusionScheduler scheduler(PNDMScheduler scheduler, MetalBinding metalBinding) {
        Objects.requireNonNull(metalBinding, "metalBinding must not be null");
        return scheduler(scheduler, metalDefaultTensorFloat32Bridge(metalBinding));
    }

    public static DiffusionDenoiser denoiser(UNetModel model, TensorAccelBridge tensorBridge) {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(tensorBridge, "tensorBridge must not be null");
        return new DiffusionDenoiser() {
            @Override
            public Tensor predict(Tensor latents, Tensor conditioning, int timestep) {
                try (AccelTensor accelLatents = tensorBridge.toAccelTensor(latents);
                        AccelTensor accelConditioning = tensorBridge.toAccelTensor(conditioning);
                        AccelTensor accelPrediction = model.predict(accelLatents, timestep, accelConditioning)) {
                    return tensorBridge.toCoreTensor(accelPrediction);
                }
            }
        };
    }

    public static TensorAccelBridge float32Bridge(
            CoreTensorReader tensorReader,
            CoreTensorFactory tensorFactory) {
        Objects.requireNonNull(tensorReader, "tensorReader must not be null");
        Objects.requireNonNull(tensorFactory, "tensorFactory must not be null");
        return new TensorAccelBridge() {
            @Override
            public AccelTensor toAccelTensor(Tensor tensor) {
                float[] values = tensorReader.readFloat32(tensor);
                return AccelTensor.fromFloatArray(values, tensor.shape().dims());
            }

            @Override
            public Tensor toCoreTensor(AccelTensor tensor) {
                return tensorFactory.createFloat32(tensor.toFloatArray(), tensor.shape());
            }
        };
    }

    public static AcceleratorAwareTensorAccelBridge metalFloat32Bridge(
            MetalBinding metalBinding,
            CoreTensorReader tensorReader,
            CoreTensorFactory tensorFactory) {
        Objects.requireNonNull(metalBinding, "metalBinding must not be null");
        TensorAccelBridge delegate = float32Bridge(tensorReader, tensorFactory);
        Map<String, Object> metadata = Map.of(
                "accelerator", "metal",
                "nativeAvailable", metalBinding.isNativeAvailable(),
                "deviceName", metalBinding.deviceName(),
                "unifiedMemory", metalBinding.isUnifiedMemory(),
                "availableMemoryBytes", metalBinding.availableMemory());
        return new AcceleratorAwareTensorAccelBridge() {
            @Override
            public AccelTensor toAccelTensor(Tensor tensor) {
                return delegate.toAccelTensor(tensor);
            }

            @Override
            public Tensor toCoreTensor(AccelTensor tensor) {
                return delegate.toCoreTensor(tensor);
            }

            @Override
            public Map<String, Object> acceleratorMetadata() {
                return metadata;
            }
        };
    }

    public static TensorAccelBridge defaultTensorFloat32Bridge(
            ComputeBackend backend,
            DeviceType deviceType,
            CoreTensorReader fallbackTensorReader) {
        Objects.requireNonNull(backend, "backend must not be null");
        Objects.requireNonNull(deviceType, "deviceType must not be null");
        Objects.requireNonNull(fallbackTensorReader, "fallbackTensorReader must not be null");
        return new TensorAccelBridge() {
            @Override
            public AccelTensor toAccelTensor(Tensor tensor) {
                return AccelTensor.fromFloatArray(readFloat32(tensor, fallbackTensorReader), tensor.shape().dims());
            }

            @Override
            public Tensor toCoreTensor(AccelTensor tensor) {
                return createDefaultTensor(tensor.toFloatArray(), tensor.shape(), backend, deviceType);
            }
        };
    }

    public static AcceleratorAwareTensorAccelBridge metalDefaultTensorFloat32Bridge(
            MetalBinding metalBinding,
            ComputeBackend backend,
            CoreTensorReader fallbackTensorReader) {
        Objects.requireNonNull(metalBinding, "metalBinding must not be null");
        TensorAccelBridge delegate = defaultTensorFloat32Bridge(backend, DeviceType.METAL, fallbackTensorReader);
        Map<String, Object> metadata = Map.of(
                "accelerator", "metal",
                "materializer", "default-tensor-f32",
                "nativeAvailable", metalBinding.isNativeAvailable(),
                "deviceName", metalBinding.deviceName(),
                "unifiedMemory", metalBinding.isUnifiedMemory(),
                "availableMemoryBytes", metalBinding.availableMemory());
        return new AcceleratorAwareTensorAccelBridge() {
            @Override
            public AccelTensor toAccelTensor(Tensor tensor) {
                return delegate.toAccelTensor(tensor);
            }

            @Override
            public Tensor toCoreTensor(AccelTensor tensor) {
                return delegate.toCoreTensor(tensor);
            }

            @Override
            public Map<String, Object> acceleratorMetadata() {
                return metadata;
            }
        };
    }

    /**
     * Convenience Metal bridge for the common case where OPD integration only
     * needs tensor materialization and accelerator metadata, not a fully
     * operational backend implementation for downstream math.
     *
     * <p>Core tensors produced by this bridge are tagged with a
     * materialization-only backend and should be treated as transport tensors
     * for adapter boundaries. Math operations on those tensors will throw until
     * the caller replaces the backend with a full execution backend.
     *
     * <p>Reference:
     * Quanhao Li et al., "DiffusionOPD: A Unified Perspective of On-Policy
     * Distillation in Diffusion Models", arXiv:2605.15055, 2026.
     */
    public static AcceleratorAwareTensorAccelBridge metalDefaultTensorFloat32Bridge(
            MetalBinding metalBinding) {
        Objects.requireNonNull(metalBinding, "metalBinding must not be null");
        return metalDefaultTensorFloat32Bridge(
                metalBinding,
                MATERIALIZATION_ONLY_BACKEND,
                StableDiffusionRunnerAdapters::strictDefaultTensorReader);
    }

    public static Map<String, Object> schedulerMetadata(PNDMScheduler scheduler) {
        Objects.requireNonNull(scheduler, "scheduler must not be null");
        return Map.of(
                "family", scheduler.family(),
                "stepCount", scheduler.stepCount(),
                "trainingTimestepCount", scheduler.trainingTimestepCount(),
                "timesteps", scheduler.timestepsArray(),
                "alphaCumprodLength", scheduler.alphasCumprod().length);
    }

    /**
     * Backend-provided conversion seam between core training tensors and the
     * safetensor-native runtime tensors.
     */
    public interface TensorAccelBridge {

        AccelTensor toAccelTensor(Tensor tensor);

        Tensor toCoreTensor(AccelTensor tensor);
    }

    public interface AcceleratorAwareTensorAccelBridge extends TensorAccelBridge {

        Map<String, Object> acceleratorMetadata();
    }

    @FunctionalInterface
    public interface CoreTensorReader {

        float[] readFloat32(Tensor tensor);
    }

    @FunctionalInterface
    public interface CoreTensorFactory {

        Tensor createFloat32(float[] values, long... shape);
    }

    private static float[] readFloat32(Tensor tensor, CoreTensorReader fallbackTensorReader) {
        if (tensor instanceof DefaultTensor defaultTensor && defaultTensor.dtype() == DType.F32) {
            long elementCount = defaultTensor.shape().numel();
            float[] values = new float[(int) elementCount];
            java.lang.foreign.MemorySegment.copy(
                    defaultTensor.buffer().segment(),
                    0,
                    java.lang.foreign.MemorySegment.ofArray(values),
                    0,
                    elementCount * Float.BYTES);
            return values;
        }
        return fallbackTensorReader.readFloat32(tensor);
    }

    private static float[] strictDefaultTensorReader(Tensor tensor) {
        if (tensor instanceof DefaultTensor defaultTensor && defaultTensor.dtype() == DType.F32) {
            long elementCount = defaultTensor.shape().numel();
            float[] values = new float[(int) elementCount];
            java.lang.foreign.MemorySegment.copy(
                    defaultTensor.buffer().segment(),
                    0,
                    java.lang.foreign.MemorySegment.ofArray(values),
                    0,
                    elementCount * Float.BYTES);
            return values;
        }
        throw new IllegalArgumentException(
                "Expected a float32 DefaultTensor for bridge materialization, got "
                        + tensor.getClass().getName()
                        + " dtype="
                        + tensor.dtype());
    }

    private static Tensor createDefaultTensor(
            float[] values,
            long[] shape,
            ComputeBackend backend,
            DeviceType deviceType) {
        CpuBuffer buffer = new CpuBuffer((long) values.length * Float.BYTES);
        buffer.segment().copyFrom(java.lang.foreign.MemorySegment.ofArray(values));
        return new DefaultTensor(
                new Shape(shape),
                DType.F32,
                deviceType,
                buffer,
                backend);
    }

    private static final class MaterializationOnlyBackend implements ComputeBackend {
        private final String name;

        private MaterializationOnlyBackend(String name) {
            this.name = name;
        }

        @Override
        public Tensor add(Tensor a, Tensor b) {
            throw unsupported();
        }

        @Override
        public Tensor sub(Tensor a, Tensor b) {
            throw unsupported();
        }

        @Override
        public Tensor mul(Tensor a, float scalar) {
            throw unsupported();
        }

        @Override
        public Tensor div(Tensor a, float scalar) {
            throw unsupported();
        }

        @Override
        public Tensor matmul(Tensor a, Tensor b) {
            throw unsupported();
        }

        @Override
        public Tensor reshape(Tensor a, long... newShape) {
            throw unsupported();
        }

        @Override
        public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
            throw unsupported();
        }

        @Override
        public java.util.List<Tensor> split(Tensor a, int axis, int parts) {
            throw unsupported();
        }

        @Override
        public Tensor attention(Tensor Q, Tensor K, Tensor V) {
            throw unsupported();
        }

        @Override
        public Tensor softmax(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor pow(Tensor a, float exponent) {
            throw unsupported();
        }

        @Override
        public Tensor mean(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor abs(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor crossEntropy(Tensor pred, Tensor target) {
            throw unsupported();
        }

        @Override
        public Tensor binaryCrossEntropy(Tensor pred, Tensor target) {
            throw unsupported();
        }

        @Override
        public Tensor cast(Tensor a, DType dtype) {
            throw unsupported();
        }

        @Override
        public Tensor to(Tensor a, DeviceType device) {
            throw unsupported();
        }

        @Override
        public Tensor mul(Tensor a, Tensor b) {
            throw unsupported();
        }

        @Override
        public Tensor div(Tensor a, Tensor b) {
            throw unsupported();
        }

        @Override
        public Tensor addScalar(Tensor a, float scalar) {
            throw unsupported();
        }

        @Override
        public Tensor zerosLike(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor sqrt(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor relu(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor sigmoid(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor tanh(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor log(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor exp(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor silu(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor flatten(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor unsqueeze(Tensor a, int dim) {
            throw unsupported();
        }

        @Override
        public Tensor squeeze(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor transpose(Tensor a) {
            throw unsupported();
        }

        @Override
        public Tensor transpose(Tensor a, int dim0, int dim1) {
            throw unsupported();
        }

        @Override
        public long numel(Tensor a) {
            return a.numel();
        }

        private UnsupportedOperationException unsupported() {
            return new UnsupportedOperationException(
                    "Backend '" + name + "' only tags bridge-materialized tensors; "
                            + "attach a real execution backend before invoking math operations.");
        }
    }
}
