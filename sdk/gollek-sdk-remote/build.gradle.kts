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
    api(project(":sdk:gollek-sdk-core"))
    api(project(":spi:gollek-spi"))
    api(project(":spi:gollek-spi-inference"))
    api(project(":spi:gollek-spi-model"))
    api(project(":spi:gollek-spi-multimodal"))
    api(project(":spi:gollek-spi-provider"))
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    implementation(group = "io.reactivex.rxjava3", name = "rxjava", version = "3.1.8")
    implementation(group = "jakarta.validation", name = "jakarta.validation-api")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.mockito", name = "mockito-core")
    testImplementation(group = "org.mockito", name = "mockito-junit-jupiter")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher")
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

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Automatic-Module-Name" to "tech.kayys.gollek.sdk.java"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
