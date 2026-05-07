rootProject.name = "gollek-engine"

include("core:gollek-tensor")
include("core:gollek-error-code")
include("core:gollek-core")
include("core:gollek-ir")
include("core:gollek-distributed")
include("core:gollek-multitenancy")
include("core:gollek-provider-core")
include("core:gollek-tokenizer-core")
include("core:gollek-model-repository")
include("core:gollek-model-database")
include("core:gollek-adapter")


include("spi:gollek-spi")
include("spi:gollek-spi-inference")
include("spi:gollek-spi-multimodal")
include("spi:gollek-spi-provider")
include("spi:gollek-spi-model")
include("spi:gollek-spi-plugin")
include("spi:gollek-spi-runtime")

include("runner:safetensor:gollek-safetensor-core")
include("runner:safetensor:gollek-safetensor-quantization")
include("runner:safetensor:gollek-safetensor-spi")
include("runner:safetensor:gollek-safetensor-api")
include("runner:safetensor:gollek-safetensor-loader")
include("runner:gguf:gollek-gguf-core")
include("runner:gguf:gollek-gguf-core")
include("runner:gguf:gollek-gguf-converter")
include("runner:gguf:gollek-gguf-core")
include("runner:gollek-diffusion")

include("core:plugin:gollek-plugin-kernel-core")

include("compiler:gollek-compiler")
include("compiler:gollek-neural-compiler")

include("training:gollek-train")
include("training:gollek-train-api")
include("training:gollek-train-data")
include("training:gollek-train-estimator")
include("training:gollek-train-transformer")
include("training:gollek-serializer")

include("training:gollek-train-examples")

include("runtime:gollek-runtime")
include("runtime:gollek-runtime-distributed")

include("backend:cpu:gollek-backend-cpu")
include("backend:cuda:gollek-backend-cuda")
include("backend:metal:gollek-backend-metal")
include("backend:metal:gollek-mlx-binding")

include("quantizer:gollek-quantizer-gptq")
include("quantizer:gollek-quantizer-awq")
include("quantizer:gollek-quantizer-autoround")
include("quantizer:gollek-quantizer-turboquant")
include("quantizer:gollek-quantizer-quip")



include("models:gollek-model-repo-hf")
include("models:gollek-model-repo-kaggle")
include("models:gollek-model-repo-local")

include("sdk:gollek-sdk")
include("sdk:gollek-sdk-api")
include("sdk:gollek-sdk-core")
include("sdk:gollek-sdk-session")
include("sdk:gollek-sdk-local")

include("ui:gollek-cli")
include("ui:gollek-api")

include("gollek-utils")

