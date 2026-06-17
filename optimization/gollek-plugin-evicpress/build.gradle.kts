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
    implementation("io.smallrye.reactive:mutiny:2.5.5")


    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")

    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")
    
    implementation(group = "io.quarkus", name = "quarkus-rest")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
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
