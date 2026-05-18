package tech.kayys.gollek.train.examples;

import tech.kayys.gollek.train.diffusion.opd.DiffusionOpdReportInspectorSupport;

/**
 * Thin example entrypoint for the shared DiffusionOPD report inspector.
 */
public final class DiffusionOpdReportInspectorExample {

    private DiffusionOpdReportInspectorExample() {
    }

    public static void main(String[] args) {
        DiffusionOpdReportInspectorSupport.runCli(args, "DiffusionOPD report inspector");
    }
}
