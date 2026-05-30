package tech.kayys.gollek.train.diffusion.api;

import java.util.List;

/**
 * Typed decision collection for workflow campaign receipt acknowledgements.
 */
public record DiffusionOpdCampaignAcknowledgementDecisions(
        DiffusionOpdCampaignReceiptAcknowledgements receiptAcknowledgements,
        List<DiffusionOpdCampaignAcknowledgementDecision> decisions,
        int decisionCount,
        int approvedCount,
        int deferredCount,
        int rejectedCount,
        boolean followUpRequired,
        String primaryDestination,
        String overallDeliveryStatus,
        String summaryMessage) {
}
