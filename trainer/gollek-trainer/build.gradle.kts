plugins {
    java
}

sourceSets {
    named("main") {
        java.include("tech/kayys/gollek/trainer/**")
    }
    named("test") {
        java.include("tech/kayys/gollek/trainer/**")
    }
}

dependencies {
    implementation(project(":trainer:gollek-trainer-api"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
}

tasks.test {
    useJUnitPlatform()
}

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Automatic-Module-Name" to "tech.kayys.gollek.trainer"
            )
        )
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
