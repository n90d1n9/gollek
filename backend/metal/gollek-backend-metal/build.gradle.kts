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

sourceSets {
    named("main") {
        java {
            setSrcDirs(listOf("src/main/java"))
            // Keep Gradle on the maintained FFM binding surface first.
            // The older MetalComputeBackend adapter layer still targets a stale
            // tensor SPI and should migrate separately from the CLI path.
            include("tech/kayys/gollek/backend/metal/binding/MetalBinding.java")
            include("tech/kayys/gollek/backend/metal/binding/MetalCpuFallback.java")
            include("tech/kayys/gollek/backend/metal/binding/MetalFlashAttentionBinding.java")
            include("tech/kayys/gollek/backend/metal/binding/MetalFlashAttentionCpuFallback.java")
            include("tech/kayys/gollek/backend/metal/binding/MetalLibraryDiscovery.java")
        }
        resources {
            srcDir("src/main/cpp/resources")
        }
    }
}

dependencies {
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
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
