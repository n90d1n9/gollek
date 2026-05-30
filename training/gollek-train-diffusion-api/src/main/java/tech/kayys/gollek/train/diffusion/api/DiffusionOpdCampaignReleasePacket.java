package tech.kayys.gollek.train.diffusion.api;

/**
 * Typed final release handoff packet for a campaign.
 */
public record DiffusionOpdCampaignReleasePacket(
        DiffusionOpdCampaignReleaseDecision releaseDecision,
        String dispatchTarget,
        String packetOutcome,
        String fallbackPacketOutcome,
        boolean readyForEngineHandoff,
        boolean reviewEscalationOpen,
        int runnableStepCount,
        int blockedEntryCount,
        String packetKey,
        String summaryMessage) {
}
