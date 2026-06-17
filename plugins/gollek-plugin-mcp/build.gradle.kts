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
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(group = "io.quarkus", name = "quarkus-reactive-routes")
    implementation(group = "io.quarkus", name = "quarkus-websockets-client")
    implementation(group = "io.smallrye.reactive", name = "smallrye-mutiny-vertx-web-client")
    implementation(group = "io.quarkus", name = "quarkus-jackson")
    implementation(group = "com.networknt", name = "json-schema-validator", version = "1.0.87")
    implementation(group = "org.zeroturnaround", name = "zt-exec", version = "1.12")
    implementation(group = "com.github.ben-manes.caffeine", name = "caffeine")
    testImplementation(group = "io.quarkus", name = "quarkus-junit-mockito")
}

sourceSets {
    main {
        java {
            exclude("tech/kayys/gollek/mcp/provider/MCPProviderConfiguration.java")
        }
    }
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
