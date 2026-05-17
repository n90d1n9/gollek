plugins {
    java
}

sourceSets {
    named("main") {
        java.exclude("**/base/**")
    }
}

dependencies {
    implementation(project(":ml:gollek-ml-core"))
    implementation(project(":ml:gollek-ml-estimator"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
