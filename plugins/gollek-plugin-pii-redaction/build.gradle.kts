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
    implementation(project(":spi:gollek-spi"))
    
    implementation(group = "jakarta.enterprise", name = "jakarta.enterprise.cdi-api")
    implementation(group = "jakarta.inject", name = "jakarta.inject-api")
    implementation(group = "jakarta.annotation", name = "jakarta.annotation-api")
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.mockito", name = "mockito-core")
    testImplementation(group = "org.mockito", name = "mockito-junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core")
    testImplementation(group = "io.quarkus", name = "quarkus-junit5")
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
