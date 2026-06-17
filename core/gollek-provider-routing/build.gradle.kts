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
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(group = "io.quarkus", name = "quarkus-mutiny")
    implementation(group = "io.quarkus", name = "quarkus-vertx")
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    implementation(group = "org.reactivestreams", name = "reactive-streams")
    implementation(group = "io.quarkus", name = "quarkus-rest-client")
    implementation(group = "io.quarkus", name = "quarkus-rest-client-jackson")
    implementation(group = "com.fasterxml.jackson.dataformat", name = "jackson-dataformat-yaml")
    implementation(group = "io.quarkus", name = "quarkus-smallrye-fault-tolerance")
    implementation(group = "io.quarkus", name = "quarkus-micrometer-registry-prometheus")
    implementation(group = "com.github.ben-manes.caffeine", name = "caffeine")
    implementation(group = "org.apache.commons", name = "commons-lang3", version = "3.17.0")
    implementation(group = "org.apache.commons", name = "commons-collections4")
    implementation(group = "jakarta.validation", name = "jakarta.validation-api")
    implementation(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "io.quarkus", name = "quarkus-junit")
    testImplementation(group = "io.rest-assured", name = "rest-assured")
    testImplementation(group = "org.assertj", name = "assertj-core")
    testImplementation(group = "org.mockito", name = "mockito-core")
    testImplementation(group = "org.awaitility", name = "awaitility")
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
