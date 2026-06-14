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
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi"))
    implementation(group = "jakarta.enterprise", name = "jakarta.enterprise.cdi-api")
    implementation(group = "jakarta.inject", name = "jakarta.inject-api")
    implementation(group = "jakarta.annotation", name = "jakarta.annotation-api")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.mockito", name = "mockito-core")
    testImplementation(group = "org.mockito", name = "mockito-junit-jupiter")
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
