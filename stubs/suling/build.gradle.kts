plugins {
    `java-library`
}

group = "tech.kayys.gollek"
version = rootProject.version

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
}

tasks.jar {
    archiveBaseName.set("gollek-suling-fallback")
    manifest {
        attributes("Automatic-Module-Name" to "tech.kayys.suling")
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
