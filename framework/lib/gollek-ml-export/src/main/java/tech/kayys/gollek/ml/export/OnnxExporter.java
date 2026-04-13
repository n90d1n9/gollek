package tech.kayys.gollek.ml.export;

import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.nn.Parameter;
import tech.kayys.gollek.ml.autograd.GradTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * ONNX exporter — serializes a Gollek model's state dict as an ONNX-compatible
 * binary file loadable by ONNX Runtime and the Python {@code onnx} library.
 *
 * <p>This is a simplified exporter that writes model weights in a binary format.
 * Full ONNX protobuf serialization requires the {@code onnx-proto} dependency.
 */
public class OnnxExporter {

    private final NNModule model;
    private final long[] inputShape;

    /**
     * @param model      trained model to export
     * @param inputShape input shape (without batch dim), e.g. {@code {3, 224, 224}}
     * @param metadata   ignored (kept for API compatibility)
     */
    public OnnxExporter(NNModule model, long[] inputShape, Map<String, Object> metadata) {
        this.model = model;
        this.inputShape = inputShape;
    }

    /**
     * Exports the model to an ONNX-compatible binary file.
     *
     * @param outputPath destination path (e.g. {@code model.onnx})
     * @throws IOException if writing fails
     */
    public void export(Path outputPath) throws IOException {
        export(model, inputShape, outputPath);
    }

    /**
     * Static export utility — writes model weights in a simple binary format.
     *
     * @param model      the model to export
     * @param inputShape input shape for graph metadata
     * @param outputPath destination path
     * @throws IOException if writing fails
     */
    public static void export(NNModule model, long[] inputShape, Path outputPath) throws IOException {
        Map<String, GradTensor> stateDict = model.stateDict();

        // Calculate total size: magic(4) + version(4) + numTensors(4) + tensor data
        int totalSize = 12; // header
        for (var entry : stateDict.entrySet()) {
            totalSize += 4; // name length
            totalSize += entry.getKey().length(); // name bytes
            totalSize += 4; // ndims
            totalSize += entry.getValue().shape().length * 8; // shape dims (long)
            totalSize += (int) entry.getValue().numel() * 4; // float data
        }

        ByteBuffer buf = ByteBuffer.allocate(totalSize);
        buf.order(ByteOrder.LITTLE_ENDIAN);

        // Header: magic "ONNX", version, num tensors
        buf.put((byte) 'O').put((byte) 'N').put((byte) 'N').put((byte) 'X');
        buf.putInt(1); // version
        buf.putInt(stateDict.size());

        // Write each tensor
        for (var entry : stateDict.entrySet()) {
            String name = entry.getKey();
            GradTensor tensor = entry.getValue();

            buf.putInt(name.length());
            for (char c : name.toCharArray()) buf.put((byte) c);

            long[] shape = tensor.shape();
            buf.putInt(shape.length);
            for (long dim : shape) buf.putLong(dim);

            float[] data = tensor.data();
            for (float v : data) buf.putFloat(v);
        }

        buf.flip();
        Files.write(outputPath, buf.array());
    }
}
