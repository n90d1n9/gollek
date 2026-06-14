import os

torch_tensor_path = './runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/core/TorchTensor.java'
with open(torch_tensor_path, 'r') as f:
    content = f.read()

content = content.replace('implements Tensor {', 'implements Tensor, AutoCloseable {')
content = content.replace('public long[] shape()', 'public long[] shapeArray()') # Rename existing to shapeArray
content = content.replace('public class TorchTensor implements Tensor, AutoCloseable {', '''public class TorchTensor implements Tensor, AutoCloseable {
    @Override
    public tech.kayys.gollek.core.tensor.Shape shape() {
        return new tech.kayys.gollek.core.tensor.Shape(shapeArray());
    }

    @Override
    public Tensor transpose(int dim0, int dim1) {
        // Simple stub for now
        return this;
    }
''')

# Fix ExecutionContext
content = content.replace('tech.kayys.gollek.runtime.tensor.ExecutionContext', 'tech.kayys.gollek.core.graph.ExecutionContext')

with open(torch_tensor_path, 'w') as f:
    f.write(content)

backend_path = './runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/core/LibTorchBackend.java'
with open(backend_path, 'r') as f:
    content = f.read()

content = content.replace('implements Backend', 'implements tech.kayys.gollek.core.backend.ComputeBackend')
content = content.replace('import tech.kayys.gollek.core.backend.ComputeBackend;', '')
content = content.replace('tech.kayys.gollek.runtime.tensor.ExecutionContext', 'tech.kayys.gollek.core.graph.ExecutionContext')

with open(backend_path, 'w') as f:
    f.write(content)

