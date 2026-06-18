plugins {
    java
}

dependencies {
    implementation(project(":runner:safetensor:gollek-safetensor-api"))
    implementation(project(":runner:safetensor:gollek-safetensor-core"))
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation("io.quarkus:quarkus-core")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
    
    // We will add quantizers if they exist, but if not we can omit them
    // Let's add them since we just included them in settings.gradle.kts
    implementation("tech.kayys.tafkir:tafkir-quantizer-gptq:0.1.0-SNAPSHOT")
    implementation("tech.kayys.tafkir:tafkir-quantizer-awq:0.1.0-SNAPSHOT")
    implementation("tech.kayys.tafkir:tafkir-quantizer-autoround:0.1.0-SNAPSHOT")
    implementation("tech.kayys.tafkir:tafkir-quantizer-turboquant:0.1.0-SNAPSHOT")
    // quantization tests use awaitility
    testImplementation("org.awaitility:awaitility:4.2.1")
    testImplementation("io.quarkus:quarkus-junit5")
}

tasks.test {
    useJUnitPlatform()
}

tasks.named("compileJava") {
    // The root build adds generated Jandex output to every source set. On a full
    // clean install Gradle can schedule this module as soon as safetensor-core
    // compileJava finishes, before its augmented classes output is complete.
    dependsOn(":runner:safetensor:gollek-safetensor-core:classes")
}
