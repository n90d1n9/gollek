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
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":ml:gollek-ml-autograd"))
    implementation(project(":ml:gollek-ml-nn"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(group = "org.slf4j", name = "slf4j-api", version = "2.0.13")
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
                "Automatic-Module-Name" to "tech.kayys.gollek.lib.multimodal"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
