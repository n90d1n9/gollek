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
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
    testRuntimeOnly(group = "org.jboss.logging", name = "jboss-logging")
    testRuntimeOnly("org.slf4j:slf4j-simple:2.0.12")
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
                "Plugin-Id" to "litert-runner",
                "Plugin-Type" to "runner",
                "Plugin-Provider" to "tech.kayys.gollek.plugin.runner.litert.LiteRTRunnerPlugin",
                "Plugin-Version" to "0.1.0-SNAPSHOT",
                "Supported-Formats" to ".litertlm,.tflite,.tfl,.task"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
