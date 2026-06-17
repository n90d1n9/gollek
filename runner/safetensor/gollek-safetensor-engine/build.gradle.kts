plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

sourceSets {
    named("main") {
        java {
            // Match the maintained Maven slice first; these legacy packages still
            // rely on older provider/runtime surfaces and migrate separately.
            exclude("tech/kayys/gollek/safetensor/config/SafetensorProviderConfig.java")
            exclude("tech/kayys/gollek/safetensor/inference/**")
            exclude("tech/kayys/gollek/safetensor/engine/lifecycle/**")
            exclude("tech/kayys/gollek/safetensor/mask/**")
        }
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation(platform("io.quarkus.platform:quarkus-bom:$quarkusVersion"))

    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")

    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.gollek:gollek-safetensor-audio:0.1.0-SNAPSHOT")

    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-core:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("io.smallrye.config:smallrye-config:3.10.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-smallrye-health")
    implementation("io.quarkus:quarkus-jackson")
    implementation("jakarta.annotation:jakarta.annotation-api:2.1.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test> {
    useJUnitPlatform()
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
