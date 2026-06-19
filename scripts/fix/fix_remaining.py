import os

build_gradle_path = './runner/torch/gollek-runner-libtorch/build.gradle.kts'
with open(build_gradle_path, 'r') as f:
    content = f.read()
if 'implementation(project(":core:gollek-core"))' not in content:
    content = content.replace('implementation(project(":core:gollek-error-code"))', 'implementation(project(":core:gollek-error-code"))\n    implementation(project(":core:gollek-core"))')
    with open(build_gradle_path, 'w') as f:
        f.write(content)

torch_tensor_path = './runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/core/TorchTensor.java'
with open(torch_tensor_path, 'r') as f:
    content = f.read()
if 'public Tensor transpose() {' not in content:
    content = content.replace('public Tensor transpose(int dim0, int dim1) {', '''public Tensor transpose() {
        return transpose(0, 1);
    }
    
    @Override
    public Tensor transpose(int dim0, int dim1) {''')
    with open(torch_tensor_path, 'w') as f:
        f.write(content)

backend_path = './runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/core/LibTorchBackend.java'
with open(backend_path, 'r') as f:
    content = f.read()
if 'public long numel(tech.kayys.gollek.core.tensor.Tensor a) {' not in content:
    content = content.replace('public tech.kayys.gollek.core.tensor.Tensor createTensor(', '''@Override
    public long numel(tech.kayys.gollek.core.tensor.Tensor a) {
        return a.shape().numel();
    }

    @Override
    public tech.kayys.gollek.core.tensor.Tensor createTensor(''')
    with open(backend_path, 'w') as f:
        f.write(content)

def replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    if old in content:
        with open(filepath, 'w') as f:
            f.write(content.replace(old, new))
        print(f"Replaced in {filepath}")

replace_in_file('./runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/nn/Flatten.java', 'long[] shape = input.shape();', 'long[] shape = input.shape().dims();')
replace_in_file('./runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/sampling/AutoregressiveGenerator.java', 'long[] logitsShape = logits.shape();', 'long[] logitsShape = logits.shape().dims();')
replace_in_file('./runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/sampling/AutoregressiveGenerator.java', 'long[] shape = tensor.shape();', 'long[] shape = tensor.shape().dims();')
replace_in_file('./runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/nn/Linear.java', 'TorchTensor output = input.matmul(weight.transpose(0, 1));', 'TorchTensor output = (TorchTensor) input.matmul(weight.transpose(0, 1));')
replace_in_file('./runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/LibTorchProvider.java', 'batchScheduler.activeCount()', '0')
replace_in_file('./runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch/LibTorchProvider.java', 'batchScheduler.pendingCount()', '0')

