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
    implementation(project(":ml:gollek-ml-autograd"))
    implementation(project(":ml:gollek-ml-nn"))
    implementation(project(":runner:onnx:gollek-ml-export-onnx"))
    implementation(project(":runner:gguf:gollek-gguf-core"))
    implementation(project(":runner:litert:gollek-litert-core"))
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
                "Automatic-Module-Name" to "tech.kayys.gollek.ml.export"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
