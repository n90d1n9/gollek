plugins {
    java
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            artifactId = "gollek-sdk-java-local"
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

dependencies {
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":core:gollek-core"))
    implementation(project(":core:gollek-model-repository"))
    implementation(project(":core:gollek-observability"))
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.gollek:gollek-engine:0.1.0-SNAPSHOT")
    implementation(project(":plugins:gollek-plugin-mcp"))
    implementation(project(":runner:gguf:gollek-gguf-converter"))
    implementation("tech.kayys.aljabr:aljabr-model-repo-hf:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-model-repo-local:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-tokenizer-core:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation(project(":runner:gguf:gollek-gguf-core"))
    implementation(project(":runner:safetensor:gollek-safetensor-core"))
    implementation(project(":runner:safetensor:gollek-safetensor-engine"))
    implementation(project(":runner:safetensor:gollek-safetensor-spi"))
    implementation(project(":core:gollek-runtime-config"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))

    
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("io.opentelemetry:opentelemetry-api:1.34.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")
    implementation("org.apache.commons:commons-lang3:3.14.0")
    implementation("org.slf4j:slf4j-api:2.0.13")
    
    compileOnly("org.graalvm.sdk:nativeimage:24.1.2")
}
