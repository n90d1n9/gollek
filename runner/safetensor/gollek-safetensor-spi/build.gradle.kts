plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
}
