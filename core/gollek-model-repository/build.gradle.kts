plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-tensor"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":core:gollek-error-code"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("io.quarkus:quarkus-cache:3.15.1")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
}
