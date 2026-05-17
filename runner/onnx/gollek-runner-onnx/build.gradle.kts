plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":core:gollek-model-runner"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    implementation(group = "com.microsoft.onnxruntime", name = "onnxruntime", version = "1.17.1")
    implementation(group = "com.microsoft.onnxruntime", name = "onnxruntime_gpu", version = "1.17.1")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(project(":core:gollek-model-repo-hf"))
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
