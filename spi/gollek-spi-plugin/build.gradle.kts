plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation("org.slf4j:slf4j-api:2.0.12")
}
