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
    implementation(project(":ml:gollek-ml-nn"))
    implementation(project(":ml:gollek-ml-autograd"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":ml:gollek-ml-optimize"))
    implementation(project(":ml:gollek-ml-selection"))
    implementation(project(":ml:gollek-ml-data"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-multimodal"))
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
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
                "Automatic-Module-Name" to "tech.kayys.gollek.ml.nlp"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
