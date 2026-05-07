plugins {
    java
}

dependencies {
    implementation(project(":runner:safetensor:gollek-safetensor-api"))
    implementation(project(":runner:safetensor:gollek-safetensor-spi"))
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
}
