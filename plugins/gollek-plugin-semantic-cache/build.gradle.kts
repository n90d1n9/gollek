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
    implementation(project(":spi:gollek-spi-inference"))
    implementation(group = "tech.kayys.gollek", name = "gollek-engine")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(group = "com.github.ben-manes.caffeine", name = "caffeine", version = "3.1.8")
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    testImplementation(group = "io.quarkus", name = "quarkus-junit5")
    testImplementation(group = "org.assertj", name = "assertj-core")
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
