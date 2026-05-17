plugins {
    java
}

dependencies {
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation(project(":quantizer:gollek-quantizer-gptq"))
    implementation(project(":runner:gguf:gollek-gguf-core"))
    implementation(project(":core:gollek-tensor"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.slf4j:slf4j-api:2.0.12")
}
