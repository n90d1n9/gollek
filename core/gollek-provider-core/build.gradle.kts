plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-error-code"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":core:gollek-tensor"))
    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-vertx")
    implementation("io.quarkus:quarkus-mutiny")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
}
