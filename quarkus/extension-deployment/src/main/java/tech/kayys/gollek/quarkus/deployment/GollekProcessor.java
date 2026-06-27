package tech.kayys.gollek.quarkus.deployment;

import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.builditem.ExtensionSslNativeSupportBuildItem;

public class GollekProcessor {

    @BuildStep
    ExtensionSslNativeSupportBuildItem setup() {
        // Quarkus build step for Gollek integration
        return new ExtensionSslNativeSupportBuildItem(false);
    }
}
