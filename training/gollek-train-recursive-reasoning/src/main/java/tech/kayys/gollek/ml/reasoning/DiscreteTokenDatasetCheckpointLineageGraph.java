package tech.kayys.gollek.ml.reasoning;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Resolved ancestry graph for recursive-reasoning trainer checkpoints under one root.
 */
public record DiscreteTokenDatasetCheckpointLineageGraph(
        Path rootDir,
        List<Node> nodes,
        List<String> duplicateRunIds,
        List<String> missingParentRunIds,
        List<String> ambiguousParentRunIds,
        List<String> parentMismatchRunIds,
        List<String> cycleRunIds) {

    public DiscreteTokenDatasetCheckpointLineageGraph {
        rootDir = Objects.requireNonNull(rootDir, "rootDir must not be null");
        nodes = immutableNodes(nodes);
        duplicateRunIds = immutableTexts(duplicateRunIds, "duplicateRunId");
        missingParentRunIds = immutableTexts(missingParentRunIds, "missingParentRunId");
        ambiguousParentRunIds = immutableTexts(ambiguousParentRunIds, "ambiguousParentRunId");
        parentMismatchRunIds = immutableTexts(parentMismatchRunIds, "parentMismatchRunId");
        cycleRunIds = immutableTexts(cycleRunIds, "cycleRunId");
    }

    public static DiscreteTokenDatasetCheckpointLineageGraph fromInventory(
            DiscreteTokenDatasetTrainerCheckpointInventory inventory) {
        Objects.requireNonNull(inventory, "inventory must not be null");
        return fromCheckpoints(inventory.rootDir(), inventory.checkpoints());
    }

    public static DiscreteTokenDatasetCheckpointLineageGraph fromCheckpoints(
            Path rootDir,
            List<DiscreteTokenDatasetTrainerCheckpointSnapshot> checkpoints) {
        Objects.requireNonNull(rootDir, "rootDir must not be null");
        List<DraftNode> drafts = checkpoints == null || checkpoints.isEmpty()
                ? List.of()
                : checkpoints.stream()
                        .map(checkpoint -> DraftNode.from(
                                Objects.requireNonNull(checkpoint, "checkpoint must not be null")))
                        .sorted(DraftNode.COMPARATOR)
                        .toList();

        Map<String, List<DraftNode>> byRunId = new LinkedHashMap<>();
        for (DraftNode draft : drafts) {
            byRunId.computeIfAbsent(draft.runId(), ignored -> new ArrayList<>()).add(draft);
        }

        List<String> duplicates = byRunId.entrySet().stream()
                .filter(entry -> entry.getValue().size() > 1)
                .map(Map.Entry::getKey)
                .sorted()
                .toList();

        Map<String, List<String>> childRunIds = new LinkedHashMap<>();
        for (DraftNode draft : drafts) {
            if (draft.parentRunId() != null) {
                childRunIds.computeIfAbsent(draft.parentRunId(), ignored -> new ArrayList<>()).add(draft.runId());
            }
        }

        List<String> missingParents = drafts.stream()
                .map(DraftNode::parentRunId)
                .filter(Objects::nonNull)
                .filter(parentRunId -> !byRunId.containsKey(parentRunId))
                .distinct()
                .sorted()
                .toList();

        List<String> ambiguousParents = drafts.stream()
                .map(DraftNode::parentRunId)
                .filter(Objects::nonNull)
                .filter(parentRunId -> byRunId.getOrDefault(parentRunId, List.of()).size() > 1)
                .distinct()
                .sorted()
                .toList();
        List<String> cycles = cycleRunIds(drafts, byRunId);
        Set<String> cycleRunIdSet = new HashSet<>(cycles);
        Map<Path, List<String>> parentMismatches = parentMismatchReasons(drafts, byRunId);
        List<String> parentMismatchRunIds = drafts.stream()
                .filter(draft -> !parentMismatches.getOrDefault(draft.checkpointDir(), List.of()).isEmpty())
                .map(DraftNode::runId)
                .distinct()
                .sorted()
                .toList();

        List<Node> nodes = drafts.stream()
                .map(draft -> {
                    List<DraftNode> parents = draft.parentRunId() == null
                            ? List.of()
                            : byRunId.getOrDefault(draft.parentRunId(), List.of());
                    List<String> children = childRunIds.getOrDefault(draft.runId(), List.of()).stream()
                            .distinct()
                            .sorted()
                            .toList();
                    return draft.toNode(
                            parents.size() == 1,
                            parents.size() > 1,
                            byRunId.getOrDefault(draft.runId(), List.of()).size() > 1,
                            cycleRunIdSet.contains(draft.runId()),
                            parentMismatches.getOrDefault(draft.checkpointDir(), List.of()),
                            children);
                })
                .toList();

        return new DiscreteTokenDatasetCheckpointLineageGraph(
                rootDir,
                nodes,
                duplicates,
                missingParents,
                ambiguousParents,
                parentMismatchRunIds,
                cycles);
    }

    public int nodeCount() {
        return nodes.size();
    }

    public boolean healthy() {
        return duplicateRunIds.isEmpty()
                && missingParentRunIds.isEmpty()
                && ambiguousParentRunIds.isEmpty()
                && parentMismatchRunIds.isEmpty()
                && cycleRunIds.isEmpty();
    }

    public List<Node> roots() {
        return nodes.stream()
                .filter(Node::root)
                .toList();
    }

    public List<Node> unresolvedNodes() {
        return nodes.stream()
                .filter(node -> !node.parentResolved() || !node.parentIdentityMatched() || node.cycleMember())
                .toList();
    }

    public Map<String, Object> healthMetadata() {
        return DiscreteTokenDatasetCheckpointLineageHealth.healthMetadata(this);
    }

    public String status() {
        return DiscreteTokenDatasetCheckpointLineageHealth.status(this);
    }

    public double healthScore() {
        return DiscreteTokenDatasetCheckpointLineageHealth.healthScore(this);
    }

    public String alertLevel() {
        return DiscreteTokenDatasetCheckpointLineageHealth.alertLevel(this);
    }

    public String summaryMessage() {
        return DiscreteTokenDatasetCheckpointLineageHealth.summaryMessage(this);
    }

    public String recommendedAction() {
        return DiscreteTokenDatasetCheckpointLineageHealth.recommendedAction(this);
    }

    public Map<String, Object> healthBadge() {
        return DiscreteTokenDatasetCheckpointLineageHealth.healthBadge(this);
    }

    public List<Map<String, Object>> checks() {
        return DiscreteTokenDatasetCheckpointLineageHealth.checks(this);
    }

    public List<Map<String, Object>> failingChecks() {
        return DiscreteTokenDatasetCheckpointLineageHealth.failingChecks(this);
    }

    public int passingCheckCount() {
        return DiscreteTokenDatasetCheckpointLineageHealth.passingCheckCount(this);
    }

    public int failingCheckCount() {
        return DiscreteTokenDatasetCheckpointLineageHealth.failingCheckCount(this);
    }

    public Map<String, Object> checkSummary() {
        return DiscreteTokenDatasetCheckpointLineageHealth.checkSummary(this);
    }

    public Optional<Map<String, Object>> primaryFailingCheck() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryFailingCheck(this);
    }

    public Optional<String> primaryFailingCheckName() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryFailingCheckName(this);
    }

    public Optional<String> primaryFailingCheckType() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryFailingCheckType(this);
    }

    public Optional<String> primaryFailingCheckCode() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryFailingCheckCode(this);
    }

    public Optional<String> primaryFailingCheckSeverity() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryFailingCheckSeverity(this);
    }

    public Optional<String> primaryFailingCheckAction() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryFailingCheckAction(this);
    }

    public Optional<String> primaryFailingCheckMessage() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryFailingCheckMessage(this);
    }

    public int issueCount() {
        return DiscreteTokenDatasetCheckpointLineageHealth.issueCount(this);
    }

    public List<String> issueTypes() {
        return DiscreteTokenDatasetCheckpointLineageHealth.issueTypes(this);
    }

    public int issueDetailCount() {
        return DiscreteTokenDatasetCheckpointLineageHealth.issueDetailCount(this);
    }

    public List<String> issueCodes() {
        return DiscreteTokenDatasetCheckpointLineageHealth.issueCodes(this);
    }

    public int blockingIssueCount() {
        return DiscreteTokenDatasetCheckpointLineageHealth.blockingIssueCount(this);
    }

    public Optional<Map<String, Object>> primaryIssue() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryIssue(this);
    }

    public Optional<String> primaryIssueCode() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryIssueCode(this);
    }

    public Optional<String> primaryIssueType() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryIssueType(this);
    }

    public Optional<String> primaryIssueAction() {
        return DiscreteTokenDatasetCheckpointLineageHealth.primaryIssueAction(this);
    }

    public List<Map<String, Object>> issueDetails() {
        return DiscreteTokenDatasetCheckpointLineageHealth.issueDetails(this);
    }

    public void requireHealthy() {
        if (!healthy()) {
            throw new IllegalStateException(healthSummary());
        }
    }

    public LineageChain chainFor(DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        String runId = checkpoint.manifest().runId();
        return nodes.stream()
                .filter(node -> node.checkpointDir().equals(checkpoint.checkpointDir()))
                .findFirst()
                .map(this::chainForNode)
                .orElseGet(() -> LineageChain.missingTarget(runId, checkpoint.checkpointDir()));
    }

    public LineageChain chainForRunId(String runId) {
        runId = requireText(runId, "runId");
        List<Node> matches = nodesByRunId().getOrDefault(runId, List.of());
        if (matches.isEmpty()) {
            return LineageChain.missingTarget(runId, null);
        }
        if (matches.size() > 1) {
            return LineageChain.ambiguousTarget(runId, matches);
        }
        return chainForNode(matches.get(0));
    }

    public String summary() {
        return "checkpoint lineage graph "
                + nodeCount()
                + " node(s), "
                + roots().size()
                + " root(s), "
                + unresolvedNodes().size()
                + " unresolved, "
                + cycleRunIds.size()
                + " cyclic, healthy="
                + healthy();
    }

    public String healthSummary() {
        if (healthy()) {
            return "checkpoint lineage graph healthy";
        }
        return "checkpoint lineage graph unhealthy: "
                + String.join(", ", issueTypes());
    }

    public Map<String, Object> toMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("rootDir", rootDir.toString());
        metadata.put("nodeCount", nodeCount());
        metadata.put("rootCount", roots().size());
        metadata.put("unresolvedCount", unresolvedNodes().size());
        metadata.put("healthy", healthy());
        metadata.put("duplicateRunIds", duplicateRunIds);
        metadata.put("missingParentRunIds", missingParentRunIds);
        metadata.put("ambiguousParentRunIds", ambiguousParentRunIds);
        metadata.put("parentMismatchRunIds", parentMismatchRunIds);
        metadata.put("cycleRunIds", cycleRunIds);
        metadata.put("health", healthMetadata());
        metadata.put("summary", summary());
        metadata.put("nodes", nodes.stream()
                .map(Node::toMetadata)
                .toList());
        return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    private LineageChain chainForNode(Node target) {
        Map<Path, Node> byDir = nodesByDir();
        Map<String, List<Node>> byRunId = nodesByRunId();
        List<Node> reverseChain = new ArrayList<>();
        Set<Path> visited = new HashSet<>();
        Node current = target;
        String missingParent = null;
        String ambiguousParent = null;
        boolean cycle = false;

        while (current != null) {
            if (!visited.add(current.checkpointDir())) {
                cycle = true;
                break;
            }
            reverseChain.add(current);
            if (current.parentRunId() == null) {
                break;
            }
            List<Node> parents = byRunId.getOrDefault(current.parentRunId(), List.of());
            if (parents.isEmpty()) {
                missingParent = current.parentRunId();
                break;
            }
            if (parents.size() > 1) {
                ambiguousParent = current.parentRunId();
                break;
            }
            current = byDir.get(parents.get(0).checkpointDir());
        }

        Collections.reverse(reverseChain);
        List<String> parentMismatchRunIds = reverseChain.stream()
                .filter(node -> !node.parentIdentityMatched())
                .map(Node::runId)
                .distinct()
                .sorted()
                .toList();
        boolean complete = missingParent == null && ambiguousParent == null && !cycle && parentMismatchRunIds.isEmpty();
        return new LineageChain(
                target.runId(),
                target.checkpointDir(),
                true,
                complete,
                cycle,
                missingParent,
                ambiguousParent,
                parentMismatchRunIds,
                reverseChain);
    }

    private static List<String> cycleRunIds(
            List<DraftNode> drafts,
            Map<String, List<DraftNode>> byRunId) {
        Map<String, String> parentByRunId = new LinkedHashMap<>();
        for (DraftNode draft : drafts) {
            if (byRunId.getOrDefault(draft.runId(), List.of()).size() != 1) {
                continue;
            }
            String parentRunId = draft.parentRunId();
            if (parentRunId == null || byRunId.getOrDefault(parentRunId, List.of()).size() != 1) {
                continue;
            }
            parentByRunId.put(draft.runId(), parentRunId);
        }

        Set<String> cycles = new LinkedHashSet<>();
        for (String runId : parentByRunId.keySet()) {
            Map<String, Integer> position = new LinkedHashMap<>();
            List<String> path = new ArrayList<>();
            String current = runId;
            while (current != null && parentByRunId.containsKey(current)) {
                Integer repeatedAt = position.get(current);
                if (repeatedAt != null) {
                    cycles.addAll(path.subList(repeatedAt, path.size()));
                    break;
                }
                position.put(current, path.size());
                path.add(current);
                current = parentByRunId.get(current);
            }
        }
        return cycles.stream()
                .sorted()
                .toList();
    }

    private static Map<Path, List<String>> parentMismatchReasons(
            List<DraftNode> drafts,
            Map<String, List<DraftNode>> byRunId) {
        Map<Path, List<String>> reasons = new LinkedHashMap<>();
        for (DraftNode draft : drafts) {
            String parentRunId = draft.parentRunId();
            if (parentRunId == null) {
                continue;
            }
            List<DraftNode> parents = byRunId.getOrDefault(parentRunId, List.of());
            if (parents.size() != 1) {
                continue;
            }
            DraftNode parent = parents.get(0);
            List<String> mismatches = new ArrayList<>();
            Long expectedStep = draft.lineage().parentCheckpointStep();
            if (expectedStep != null && expectedStep.longValue() != parent.checkpointStep()) {
                mismatches.add("parent checkpointStep expected "
                        + expectedStep
                        + " but found "
                        + parent.checkpointStep());
            }
            String expectedFingerprint = draft.lineage().parentDatasetFingerprint();
            if (expectedFingerprint != null && !expectedFingerprint.equals(parent.datasetFingerprint())) {
                mismatches.add("parent dataset fingerprint expected "
                        + expectedFingerprint
                        + " but found "
                        + parent.datasetFingerprint());
            }
            if (!mismatches.isEmpty()) {
                reasons.put(draft.checkpointDir(), List.copyOf(mismatches));
            }
        }
        return Collections.unmodifiableMap(reasons);
    }

    private Map<String, List<Node>> nodesByRunId() {
        Map<String, List<Node>> byRunId = new LinkedHashMap<>();
        for (Node node : nodes) {
            byRunId.computeIfAbsent(node.runId(), ignored -> new ArrayList<>()).add(node);
        }
        return Collections.unmodifiableMap(byRunId);
    }

    private Map<Path, Node> nodesByDir() {
        Map<Path, Node> byDir = new LinkedHashMap<>();
        for (Node node : nodes) {
            byDir.put(node.checkpointDir(), node);
        }
        return Collections.unmodifiableMap(byDir);
    }

    private static List<Node> immutableNodes(List<Node> nodes) {
        if (nodes == null || nodes.isEmpty()) {
            return List.of();
        }
        return nodes.stream()
                .map(node -> Objects.requireNonNull(node, "lineage node must not be null"))
                .toList();
    }

    private static List<String> immutableTexts(List<String> values, String name) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream()
                .map(value -> requireText(value, name))
                .toList();
    }

    private static String requireText(String value, String name) {
        value = Objects.requireNonNull(value, name + " must not be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }

    private record DraftNode(
            Path checkpointDir,
            String runId,
            String parentRunId,
            long checkpointStep,
            long createdAtEpochMillis,
            String status,
            boolean ready,
            String datasetFingerprint,
            DiscreteTokenDatasetCheckpointLineage lineage) {

        static final Comparator<DraftNode> COMPARATOR = Comparator
                .comparingLong(DraftNode::createdAtEpochMillis)
                .thenComparingLong(DraftNode::checkpointStep)
                .thenComparing(DraftNode::runId)
                .thenComparing(draft -> draft.checkpointDir().toString());

        static DraftNode from(DiscreteTokenDatasetTrainerCheckpointSnapshot checkpoint) {
            DiscreteTokenDatasetCheckpointManifestSnapshot manifest = checkpoint.manifest();
            return new DraftNode(
                    checkpoint.checkpointDir(),
                    manifest.runId(),
                    manifest.lineage().parentRunId(),
                    manifest.checkpointStep(),
                    manifest.createdAtEpochMillis(),
                    checkpoint.status(),
                    checkpoint.ready(),
                    manifest.fingerprint().value(),
                    manifest.lineage());
        }

        Node toNode(
                boolean parentPresent,
                boolean parentAmbiguous,
                boolean duplicateRunId,
                boolean cycleMember,
                List<String> parentMismatchReasons,
                List<String> childRunIds) {
            return new Node(
                    checkpointDir,
                    runId,
                    parentRunId,
                    checkpointStep,
                    createdAtEpochMillis,
                    status,
                    ready,
                    lineage,
                    parentPresent,
                    parentAmbiguous,
                    duplicateRunId,
                    cycleMember,
                    parentMismatchReasons,
                    childRunIds);
        }
    }

    public record Node(
            Path checkpointDir,
            String runId,
            String parentRunId,
            long checkpointStep,
            long createdAtEpochMillis,
            String status,
            boolean ready,
            DiscreteTokenDatasetCheckpointLineage lineage,
            boolean parentPresent,
            boolean parentAmbiguous,
            boolean duplicateRunId,
            boolean cycleMember,
            List<String> parentMismatchReasons,
            List<String> childRunIds) {

        public Node {
            checkpointDir = Objects.requireNonNull(checkpointDir, "checkpointDir must not be null");
            runId = requireText(runId, "runId");
            parentRunId = parentRunId == null ? null : requireText(parentRunId, "parentRunId");
            if (checkpointStep < 0L) {
                throw new IllegalArgumentException("checkpointStep must be >= 0 but was " + checkpointStep);
            }
            if (createdAtEpochMillis < 0L) {
                throw new IllegalArgumentException("createdAtEpochMillis must be >= 0 but was " + createdAtEpochMillis);
            }
            status = requireText(status, "status");
            lineage = Objects.requireNonNull(lineage, "lineage must not be null");
            parentMismatchReasons = immutableTexts(parentMismatchReasons, "parentMismatchReason");
            childRunIds = immutableTexts(childRunIds, "childRunId");
            if (parentRunId == null && (parentPresent || parentAmbiguous)) {
                throw new IllegalArgumentException("root lineage node cannot have parent resolution flags");
            }
        }

        public boolean root() {
            return parentRunId == null;
        }

        public boolean parentResolved() {
            return parentRunId == null || (parentPresent && !parentAmbiguous);
        }

        public boolean parentIdentityMatched() {
            return parentMismatchReasons.isEmpty();
        }

        public String summary() {
            String parent = parentRunId == null ? "root" : "parent=" + parentRunId;
            return "lineage node "
                    + runId
                    + " step "
                    + checkpointStep
                    + " "
                    + parent
                    + " status="
                    + status;
        }

        public Map<String, Object> toMetadata() {
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("checkpointDir", checkpointDir.toString());
            metadata.put("runId", runId);
            metadata.put("checkpointStep", checkpointStep);
            metadata.put("createdAtEpochMillis", createdAtEpochMillis);
            metadata.put("status", status);
            metadata.put("ready", ready);
            metadata.put("root", root());
            metadata.put("parentResolved", parentResolved());
            metadata.put("parentPresent", parentPresent);
            metadata.put("parentAmbiguous", parentAmbiguous);
            metadata.put("duplicateRunId", duplicateRunId);
            metadata.put("cycleMember", cycleMember);
            metadata.put("parentIdentityMatched", parentIdentityMatched());
            metadata.put("parentMismatchReasons", parentMismatchReasons);
            if (parentRunId != null) {
                metadata.put("parentRunId", parentRunId);
            }
            metadata.put("childRunIds", childRunIds);
            metadata.put("lineage", lineage.toMetadata());
            metadata.put("summary", summary());
            return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }
    }

    public record LineageChain(
            String targetRunId,
            Path targetCheckpointDir,
            boolean targetPresent,
            boolean complete,
            boolean cycleDetected,
            String missingParentRunId,
            String ambiguousParentRunId,
            List<String> parentMismatchRunIds,
            List<Node> nodes) {

        public LineageChain {
            targetRunId = requireText(targetRunId, "targetRunId");
            missingParentRunId = missingParentRunId == null
                    ? null
                    : requireText(missingParentRunId, "missingParentRunId");
            ambiguousParentRunId = ambiguousParentRunId == null
                    ? null
                    : requireText(ambiguousParentRunId, "ambiguousParentRunId");
            parentMismatchRunIds = immutableTexts(parentMismatchRunIds, "parentMismatchRunId");
            nodes = immutableNodes(nodes);
            if (!targetPresent && !nodes.isEmpty()) {
                throw new IllegalArgumentException("missing target lineage chain must not include nodes");
            }
            if (complete && (!targetPresent
                    || cycleDetected
                    || missingParentRunId != null
                    || ambiguousParentRunId != null
                    || !parentMismatchRunIds.isEmpty())) {
                throw new IllegalArgumentException("complete lineage chain cannot have unresolved state");
            }
        }

        static LineageChain missingTarget(String runId, Path checkpointDir) {
            return new LineageChain(runId, checkpointDir, false, false, false, null, null, List.of(), List.of());
        }

        static LineageChain ambiguousTarget(String runId, List<Node> nodes) {
            return new LineageChain(runId, null, true, false, false, null, runId, List.of(), nodes);
        }

        public int depth() {
            return nodes.size();
        }

        public List<String> runIds() {
            return nodes.stream()
                    .map(Node::runId)
                    .toList();
        }

        public String summary() {
            if (!targetPresent) {
                return "lineage chain target missing: " + targetRunId;
            }
            return "lineage chain "
                    + String.join(" -> ", runIds())
                    + " complete="
                    + complete;
        }

        public Map<String, Object> toMetadata() {
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("targetRunId", targetRunId);
            if (targetCheckpointDir != null) {
                metadata.put("targetCheckpointDir", targetCheckpointDir.toString());
            }
            metadata.put("targetPresent", targetPresent);
            metadata.put("complete", complete);
            metadata.put("cycleDetected", cycleDetected);
            metadata.put("depth", depth());
            metadata.put("runIds", runIds());
            if (missingParentRunId != null) {
                metadata.put("missingParentRunId", missingParentRunId);
            }
            if (ambiguousParentRunId != null) {
                metadata.put("ambiguousParentRunId", ambiguousParentRunId);
            }
            metadata.put("parentMismatchRunIds", parentMismatchRunIds);
            metadata.put("summary", summary());
            metadata.put("nodes", nodes.stream()
                    .map(Node::toMetadata)
                    .toList());
            return Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        }
    }
}
