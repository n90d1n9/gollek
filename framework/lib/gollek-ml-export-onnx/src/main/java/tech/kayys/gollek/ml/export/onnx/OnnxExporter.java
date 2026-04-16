package tech.kayys.gollek.ml.export.onnx;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
import tech.kayys.gollek.ml.safetensors.SafetensorReader;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Exports a Gollek model's weights to a valid ONNX protobuf file
 * (IR version 8, opset 17) loadable by ONNX Runtime and the Python {@code onnx} library.
 *
 * <p>The output is a weights-only ONNX graph: all tensors are written as
 * {@code GraphProto.initializer} entries with no operator nodes. This is the
 * correct representation for a parameter store — operator nodes are added by
 * the runtime or a downstream graph builder.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * // From a .safetensors file
 * OnnxExporter.fromSafetensor(Path.of("model.safetensors"))
 *             .export(Path.of("model.onnx"));
 *
 * // With explicit shape hints
 * OnnxExporter.fromSafetensor(Path.of("model.safetensors"),
 *         Map.of("encoder.layer.0.weight", new long[]{768, 768}))
 *     .export(Path.of("model.onnx"));
 *
 * // From a trained NNModule
 * OnnxExporter.fromModel(model).export(Path.of("model.onnx"));
 * }</pre>
 */
public final class OnnxExporter {

    // ONNX ModelProto field numbers
    private static final int MODEL_IR_VERSION   = 1;
    private static final int MODEL_PRODUCER     = 2;
    private static final int MODEL_GRAPH        = 7;
    private static final int MODEL_OPSET_IMPORT = 8;
    // OperatorSetIdProto
    private static final int OPSET_VERSION      = 2;
    // GraphProto
    private static final int GRAPH_NAME         = 1;
    private static final int GRAPH_INITIALIZER  = 6;
    // TensorProto
    private static final int TENSOR_DIMS        = 1;
    private static final int TENSOR_DATA_TYPE   = 2;
    private static final int TENSOR_NAME        = 8;
    private static final int TENSOR_RAW_DATA    = 9;
    // ONNX data type: FLOAT = 1
    private static final int ONNX_FLOAT         = 1;

    private final Map<String, GradTensor> stateDict;

    private OnnxExporter(Map<String, GradTensor> stateDict) {
        this.stateDict = stateDict;
    }

    // ── Factories ────────────────────────────────────────────────────────────

    public static OnnxExporter fromModel(NNModule model) {
        return new OnnxExporter(model.stateDict());
    }

    /**
     * Loads tensors from a {@code .safetensors} file.
     * Shapes default to 1-D {@code [n]} unless overridden via {@code shapeHints}.
     */
    public static OnnxExporter fromSafetensor(Path path, Map<String, long[]> shapeHints) throws IOException {
        Map<String, float[]> raw = SafetensorReader.read(path);
        Map<String, GradTensor> dict = new LinkedHashMap<>();
        for (var e : raw.entrySet()) {
            float[] data = e.getValue();
            long[] shape = shapeHints.getOrDefault(e.getKey(), new long[]{data.length});
            dict.put(e.getKey(), GradTensor.of(data, shape));
        }
        return new OnnxExporter(dict);
    }

    public static OnnxExporter fromSafetensor(Path path) throws IOException {
        return fromSafetensor(path, Map.of());
    }

    // ── Export ───────────────────────────────────────────────────────────────

    public void export(Path outputPath) throws IOException {
        Files.write(outputPath, buildModelProto());
    }

    // ── Protobuf encoding ────────────────────────────────────────────────────

    private byte[] buildModelProto() throws IOException {
        var out = new ByteArrayOutputStream();

        // ir_version = 8
        writeVarint(out, tag(MODEL_IR_VERSION, 0));
        writeVarint(out, 8L);

        // producer_name = "gollek"
        writeString(out, MODEL_PRODUCER, "gollek");

        // opset_import { version: 17 }
        var opset = new ByteArrayOutputStream();
        writeVarint(opset, tag(OPSET_VERSION, 0));
        writeVarint(opset, 17L);
        writeBytes(out, MODEL_OPSET_IMPORT, opset.toByteArray());

        // graph
        writeBytes(out, MODEL_GRAPH, buildGraphProto());

        return out.toByteArray();
    }

    private byte[] buildGraphProto() throws IOException {
        var out = new ByteArrayOutputStream();
        writeString(out, GRAPH_NAME, "gollek_model");
        for (var e : stateDict.entrySet()) {
            writeBytes(out, GRAPH_INITIALIZER, buildTensorProto(e.getKey(), e.getValue()));
        }
        return out.toByteArray();
    }

    private byte[] buildTensorProto(String name, GradTensor tensor) throws IOException {
        var out = new ByteArrayOutputStream();

        for (long dim : tensor.shape()) {
            writeVarint(out, tag(TENSOR_DIMS, 0));
            writeVarint(out, dim);
        }

        writeVarint(out, tag(TENSOR_DATA_TYPE, 0));
        writeVarint(out, ONNX_FLOAT);

        writeString(out, TENSOR_NAME, name);

        float[] data = tensor.data();
        ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) buf.putFloat(v);
        writeBytes(out, TENSOR_RAW_DATA, buf.array());

        return out.toByteArray();
    }

    // ── Protobuf wire format ─────────────────────────────────────────────────

    private static int tag(int field, int wireType) {
        return (field << 3) | wireType;
    }

    private static void writeVarint(OutputStream out, long v) throws IOException {
        while ((v & ~0x7FL) != 0) {
            out.write((int) ((v & 0x7F) | 0x80));
            v >>>= 7;
        }
        out.write((int) v);
    }

    private static void writeString(OutputStream out, int field, String value) throws IOException {
        writeBytes(out, field, value.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    private static void writeBytes(OutputStream out, int field, byte[] value) throws IOException {
        writeVarint(out, tag(field, 2)); // wire type 2 = length-delimited
        writeVarint(out, value.length);
        out.write(value);
    }
}
