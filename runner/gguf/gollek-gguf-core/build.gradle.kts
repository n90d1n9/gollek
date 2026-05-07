plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi-model"))
    implementation("com.google.code.gson:gson:2.10.1")
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation("org.slf4j:slf4j-api:2.0.12")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
}
