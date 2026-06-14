plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi-model"))
    implementation("io.quarkus:quarkus-hibernate-reactive-panache")
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
