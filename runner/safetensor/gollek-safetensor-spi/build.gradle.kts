plugins {
    java
}

dependencies {
    implementation("tech.kayys.aljabr:aljabr-tokenizer-core:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
}
