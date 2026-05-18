///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-ml-diffusion-opd:0.1.0-SNAPSHOT

import tech.kayys.gollek.train.diffusion.opd.DiffusionOpdReportInspectorSupport;

/**
 * JBang wrapper for the shared DiffusionOPD report inspector.
 */
public class trainer_diffusion_opd_report_inspector {
    public static void main(String[] args) {
        DiffusionOpdReportInspectorSupport.runCli(
                args,
                "====================================================",
                " Gollek DiffusionOPD Report Inspector",
                "====================================================");
    }
}
