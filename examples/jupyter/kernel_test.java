///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//DEPS tech.kayys.gollek:gollek-ml-ml:0.3.0-SNAPSHOT
//COMPILE_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS local,mavencentral,github=https://maven.pkg.github.com/bhangun/gollek

import tech.kayys.gollek.ml.Gollek;

public class kernel_test {
    public static void main(String[] args) {
        System.out.println("=== Gollek Kernel Test ===");
        System.out.println("Java version: " + System.getProperty("java.version"));
        Gollek.printInfo();
        System.out.println("✓ Kernel working!");
    }
}
