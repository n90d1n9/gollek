plugins {
    java
}

dependencies {
    implementation(project(":runner:gguf:gollek-gguf-core"))
    implementation(project(":core:gollek-core"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
}
