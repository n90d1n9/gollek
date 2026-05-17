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
    compileOnly(project(":spi:gollek-spi-plugin"))
    compileOnly(project(":spi:gollek-spi"))
    compileOnly(project(":spi:gollek-spi-provider"))
    compileOnly(project(":spi:gollek-spi-inference"))
    compileOnly(group = "io.smallrye.reactive", name = "mutiny")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    compileOnly(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.mockito", name = "mockito-core")
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
                "Plugin-Class" to "tech.kayys.gollek.plugin.cloud.openai.OpenAiCloudProvider",
                "Plugin-Id" to "openai-cloud-provider",
                "Plugin-Version" to "0.1.0-SNAPSHOT"
            )
        )
    }
}

val installPluginJar by tasks.registering(Copy::class) {
    dependsOn(tasks.jar)
    from(tasks.jar)
    into(file("${System.getProperty("user.home")}/.gollek/plugins"))
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
