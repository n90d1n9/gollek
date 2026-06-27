plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-error-code"))
    implementation(project(":spi:gollek-spi"))
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0")
    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-vertx")
    implementation("io.quarkus:quarkus-mutiny")
    implementation("io.quarkus:quarkus-rest-client")
    implementation("io.quarkus:quarkus-rest-client-jackson")
    implementation("io.quarkus:quarkus-smallrye-fault-tolerance")
    implementation("io.quarkus:quarkus-micrometer-registry-prometheus")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("org.reactivestreams:reactive-streams:1.0.4")
    implementation("com.fasterxml.jackson.dataformat:jackson-dataformat-yaml:2.16.1")
    implementation("com.github.ben-manes.caffeine:caffeine:3.1.8")
    implementation("org.apache.commons:commons-lang3:3.17.0")
    implementation("org.apache.commons:commons-collections4:4.4")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.assertj:assertj-core:3.25.3")
    testImplementation("org.awaitility:awaitility:4.2.1")
    testImplementation("org.testcontainers:testcontainers:1.19.7")
    testImplementation("org.testcontainers:junit-jupiter:1.19.7")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.quarkus:quarkus-vault")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}
