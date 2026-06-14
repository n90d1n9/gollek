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
    implementation(project(":ml:gollek-ml-api"))
    implementation(project(":sdk:gollek-sdk"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(group = "dev.langchain4j", name = "langchain4j-core", version = "0.35.0")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.mockito", name = "mockito-core", version = "5.14.2")
    testImplementation(group = "org.mockito", name = "mockito-junit-jupiter", version = "5.14.2")
    testImplementation(group = "org.assertj", name = "assertj-core", version = "3.26.3")
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
