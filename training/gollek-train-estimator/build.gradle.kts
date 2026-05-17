plugins {
    java
}

sourceSets {
    named("main") {
        java.exclude("**/base/**")
        java.exclude("**/CalibratedClassifierCV.java")
    }
}

dependencies {
    implementation(project(":ml:gollek-ml-core"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
