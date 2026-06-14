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
    
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":core:gollek-observability"))
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "com.fasterxml.jackson.datatype", name = "jackson-datatype-jsr310")
    implementation(group = "jakarta.validation", name = "jakarta.validation-api")
    compileOnly(group = "jakarta.enterprise", name = "jakarta.enterprise.cdi-api")
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
