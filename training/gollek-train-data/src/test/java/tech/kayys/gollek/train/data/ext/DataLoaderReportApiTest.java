package tech.kayys.gollek.train.data.ext;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.train.data.DataLoader;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class DataLoaderReportApiTest {

    @Test
    void classificationReportComputedApiIsPublicOutsideDataPackage() {
        var report = new DataLoader.ClassificationDistributionReport(
                4,
                1,
                2,
                new int[] {3, 1},
                List.of(new int[] {3, 1}));
        var candidate = new DataLoader.ClassificationDistributionReport(
                4,
                1,
                2,
                new int[] {1, 3},
                List.of(new int[] {1, 3}));

        assertArrayEquals(new double[] {0.75, 0.25}, report.classFractions(), 1e-6);
        assertEquals(0, report.majorityClassIndex());
        assertEquals(3, report.majorityClassCount());
        assertEquals(3.0, report.imbalanceRatio(), 1e-6);

        Map<String, Object> metadata = report.toMetadata("report");
        assertEquals(2, metadata.get("report.numClasses"));

        DataLoader.ClassificationDistributionDriftReport drift = report.driftTo(candidate);
        assertEquals(0.5, drift.totalVariationDistance(), 1e-6);
        assertEquals(0, drift.maxDeltaClassIndex());
    }

    @Test
    void multiLabelReportComputedApiIsPublicOutsideDataPackage() {
        var report = new DataLoader.MultiLabelDistributionReport(
                3,
                1,
                2,
                new int[] {2, 1},
                new int[] {1, 2},
                List.of(new int[] {2, 1}));
        var candidate = new DataLoader.MultiLabelDistributionReport(
                3,
                1,
                2,
                new int[] {1, 2},
                new int[] {2, 1},
                List.of(new int[] {1, 2}));

        assertArrayEquals(new double[] {2.0 / 3.0, 1.0 / 3.0}, report.positiveFractions(), 1e-6);
        assertEquals(2.0, report.positiveImbalanceRatio(), 1e-6);
        assertEquals(1.0, report.labelCardinality(), 1e-6);
        assertEquals(0.5, report.labelDensity(), 1e-6);

        Map<String, Object> metadata = report.toMetadata("report");
        assertEquals(2, metadata.get("report.labelCount"));

        DataLoader.MultiLabelDistributionDriftReport drift = report.driftTo(candidate);
        assertEquals(1.0 / 3.0, drift.meanAbsolutePositiveFractionDelta(), 1e-6);
        assertEquals(0, drift.maxDeltaLabelIndex());
    }
}
