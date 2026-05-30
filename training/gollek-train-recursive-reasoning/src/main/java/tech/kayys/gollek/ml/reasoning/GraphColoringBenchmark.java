package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reusable graph-coloring benchmark adapter for recursive-reasoning experiments.
 */
public final class GraphColoringBenchmark {
    private GraphColoringBenchmark() {
    }

    public static GraphColoringEvaluation evaluate(
            GraphColoringProblem problem,
            GraphColoringSolution solution) {
        Objects.requireNonNull(problem, "problem must not be null");
        Objects.requireNonNull(solution, "solution must not be null");
        if (problem.nodeCount() != solution.nodeCount()) {
            throw new IllegalArgumentException(
                    "solution node count must match problem node count: "
                            + solution.nodeCount() + " vs " + problem.nodeCount());
        }
        if (problem.colorCount() != solution.colorCount()) {
            throw new IllegalArgumentException(
                    "solution color count must match problem color count: "
                            + solution.colorCount() + " vs " + problem.colorCount());
        }

        int uncoloredNodes = 0;
        int fixedViolations = 0;
        int edgeConflicts = 0;
        for (int node = 0; node < problem.nodeCount(); node++) {
            int color = solution.color(node);
            if (color == GraphColoringProblem.UNCOLORED) {
                uncoloredNodes++;
            }
            if (problem.hasFixedColor(node) && color != problem.fixedColor(node)) {
                fixedViolations++;
            }
        }

        for (GraphColoringEdge edge : problem.edges()) {
            int left = solution.color(edge.leftNode());
            int right = solution.color(edge.rightNode());
            if (left != GraphColoringProblem.UNCOLORED && left == right) {
                edgeConflicts++;
            }
        }

        int conflicts = uncoloredNodes + fixedViolations + edgeConflicts;
        return new GraphColoringEvaluation(
                uncoloredNodes == 0,
                fixedViolations == 0,
                edgeConflicts == 0,
                uncoloredNodes,
                fixedViolations,
                edgeConflicts,
                0,
                conflicts,
                Map.of(
                        "nodeCount", problem.nodeCount(),
                        "colorCount", problem.colorCount(),
                        "edgeCount", problem.edges().size(),
                        "fixedColorCount", fixedColorCount(problem)));
    }

    public static GraphColoringEvaluation evaluateTokens(
            GraphColoringProblem problem,
            int[] solutionTokens) {
        Objects.requireNonNull(problem, "problem must not be null");
        GraphColoringTokenDecodeResult decoded = GraphColoringTokenCodec.decodeSolution(
                problem.nodeCount(),
                problem.colorCount(),
                solutionTokens);
        GraphColoringEvaluation evaluation = evaluate(problem, decoded.solution());
        int conflicts = evaluation.conflictCount() + decoded.invalidTokenCount();
        return new GraphColoringEvaluation(
                evaluation.complete() && decoded.uncoloredNodeCount() == 0,
                evaluation.respectsFixedColors(),
                evaluation.edgeSafe(),
                evaluation.uncoloredNodeCount(),
                evaluation.fixedViolationCount(),
                evaluation.edgeConflictCount(),
                decoded.invalidTokenCount(),
                conflicts,
                Map.of(
                        "nodeCount", problem.nodeCount(),
                        "colorCount", problem.colorCount(),
                        "edgeCount", problem.edges().size(),
                        "fixedColorCount", fixedColorCount(problem),
                        "invalidTokenCount", decoded.invalidTokenCount(),
                        "uncoloredNodeCount", decoded.uncoloredNodeCount(),
                        "colorTokenCount", decoded.colorTokenCount()));
    }

    public static DiscreteTokenEvaluation evaluateAsDiscrete(
            GraphColoringProblem problem,
            GraphColoringSolution solution) {
        Objects.requireNonNull(solution, "solution must not be null");
        return toDiscreteEvaluation(evaluate(problem, solution), solution.canonicalKey());
    }

    public static DiscreteTokenEvaluation evaluateTokensAsDiscrete(
            GraphColoringProblem problem,
            int[] solutionTokens) {
        Objects.requireNonNull(problem, "problem must not be null");
        GraphColoringTokenDecodeResult decoded = GraphColoringTokenCodec.decodeSolution(
                problem.nodeCount(),
                problem.colorCount(),
                solutionTokens);
        return toDiscreteEvaluation(evaluateTokens(problem, solutionTokens), decoded.solution().canonicalKey());
    }

    public static GraphColoringCoverageReport coverage(
            GraphColoringProblem problem,
            List<GraphColoringSolution> candidates) {
        return coverage(problem, candidates, -1);
    }

    public static GraphColoringCoverageReport coverageAgainstAllSolutions(
            GraphColoringProblem problem,
            List<GraphColoringSolution> candidates) {
        Objects.requireNonNull(problem, "problem must not be null");
        return coverage(problem, candidates, GraphColoringSolver.count(problem));
    }

    public static GraphColoringCoverageReport coverage(
            GraphColoringProblem problem,
            List<GraphColoringSolution> candidates,
            int knownSolutionCount) {
        Objects.requireNonNull(problem, "problem must not be null");
        List<GraphColoringSolution> candidateList = List.copyOf(
                Objects.requireNonNull(candidates, "candidates must not be null"));
        List<DiscreteTokenEvaluation> evaluations = new ArrayList<>(candidateList.size());
        for (GraphColoringSolution candidate : candidateList) {
            evaluations.add(evaluateAsDiscrete(problem, candidate));
        }
        return toGraphColoringCoverageReport(DiscreteTokenCoverage.summarize(evaluations, knownSolutionCount));
    }

    public static GraphColoringCoverageReport coverageTokens(
            GraphColoringProblem problem,
            List<int[]> candidateTokens) {
        return coverageTokens(problem, candidateTokens, -1);
    }

    public static GraphColoringCoverageReport coverageTokensAgainstAllSolutions(
            GraphColoringProblem problem,
            List<int[]> candidateTokens) {
        Objects.requireNonNull(problem, "problem must not be null");
        return coverageTokens(problem, candidateTokens, GraphColoringSolver.count(problem));
    }

    public static GraphColoringCoverageReport coverageTokens(
            GraphColoringProblem problem,
            List<int[]> candidateTokens,
            int knownSolutionCount) {
        Objects.requireNonNull(problem, "problem must not be null");
        List<int[]> candidateList = List.copyOf(
                Objects.requireNonNull(candidateTokens, "candidateTokens must not be null"));
        List<DiscreteTokenEvaluation> evaluations = new ArrayList<>(candidateList.size());
        for (int[] tokens : candidateList) {
            evaluations.add(evaluateTokensAsDiscrete(problem, tokens));
        }
        return toGraphColoringCoverageReport(DiscreteTokenCoverage.summarize(evaluations, knownSolutionCount));
    }

    static GraphColoringCoverageReport toGraphColoringCoverageReport(DiscreteTokenCoverageReport report) {
        return new GraphColoringCoverageReport(
                report.candidateCount(),
                report.validCandidateCount(),
                report.uniqueValidCandidateCount(),
                report.duplicateValidCandidateCount(),
                report.knownSolutionCount(),
                report.validRate(),
                report.coverageRate());
    }

    private static DiscreteTokenEvaluation toDiscreteEvaluation(
            GraphColoringEvaluation evaluation,
            String canonicalKey) {
        return evaluation.valid()
                ? DiscreteTokenEvaluation.valid(canonicalKey, evaluation.metadata())
                : DiscreteTokenEvaluation.invalid(evaluation.conflictCount(), evaluation.metadata());
    }

    private static int fixedColorCount(GraphColoringProblem problem) {
        int count = 0;
        for (int node = 0; node < problem.nodeCount(); node++) {
            if (problem.hasFixedColor(node)) {
                count++;
            }
        }
        return count;
    }
}
