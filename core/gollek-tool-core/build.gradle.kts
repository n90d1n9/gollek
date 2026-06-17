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
    implementation(group = "io.quarkus", name = "quarkus-core")
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "org.hibernate.validator", name = "hibernate-validator", version = "8.0.1.Final")
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
