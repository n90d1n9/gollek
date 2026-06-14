import os
import glob

directory = './runner/torch/gollek-runner-libtorch/src/main/java/tech/kayys/gollek/inference/libtorch'

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = content.replace('tech.kayys.gollek.runtime.tensor.Tensor', 'tech.kayys.gollek.core.tensor.Tensor')
    new_content = new_content.replace('tech.kayys.gollek.runtime.tensor.DType', 'tech.kayys.gollek.core.tensor.DType')
    new_content = new_content.replace('tech.kayys.gollek.runtime.tensor.Device', 'tech.kayys.gollek.core.tensor.DeviceType')
    new_content = new_content.replace('tech.kayys.gollek.runtime.tensor.Backend', 'tech.kayys.gollek.core.backend.ComputeBackend')
    new_content = new_content.replace('tech.kayys.gollek.runtime.tensor.DefaultTensor', 'tech.kayys.gollek.core.tensor.Tensor') # not sure, let's remove DefaultTensor
    new_content = new_content.replace('tech.kayys.gollek.provider.core.session.', 'tech.kayys.gollek.provider.core.session.') # no change needed but wait, it might be correct
    new_content = new_content.replace('tech.kayys.gollek.runtime.inference.batch.ContinuousBatchScheduler', 'tech.kayys.gollek.runtime.batch.ContinuousBatchScheduler')
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.java'):
            process_file(os.path.join(root, file))
