package tech.kayys.gollek.ml.reasoning;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Small reusable N-Queens benchmark adapter for recursive-reasoning experiments.
 */
public final class NQueensBenchmark {
    private NQueensBenchmark() {
    }

    public static NQueensEvaluation evaluate(NQueensProblem problem, NQueensSolution solution) {
        Objects.requireNonNull(problem, "problem must not be null");
        Objects.requireNonNull(solution, "solution must not be null");
        if (problem.size() != solution.size()) {
            throw new IllegalArgumentException(
                    "solution size must match problem size: " + solution.size() + " vs " + problem.size());
        }

        int missingRows = 0;
        int fixedViolations = 0;
        int columnConflicts = 0;
        int diagonalConflicts = 0;

        for (int row = 0; row < problem.size(); row++) {
            int column = solution.column(row);
            if (column == NQueensProblem.EMPTY) {
                missingRows++;
            }
            if (problem.hasFixedQueen(row) && column != problem.fixedColumn(row)) {
                fixedViolations++;
            }
        }

        for (int leftRow = 0; leftRow < problem.size(); leftRow++) {
            int leftColumn = solution.column(leftRow);
            if (leftColumn == NQueensProblem.EMPTY) {
                continue;
            }
            for (int rightRow = leftRow + 1; rightRow < problem.size(); rightRow++) {
                int rightColumn = solution.column(rightRow);
                if (rightColumn == NQueensProblem.EMPTY) {
                    continue;
                }
                if (leftColumn == rightColumn) {
                    columnConflicts++;
                }
                if (Math.abs(leftRow - rightRow) == Math.abs(leftColumn - rightColumn)) {
                    diagonalConflicts++;
                }
            }
        }

        int conflicts = missingRows + fixedViolations + columnConflicts + diagonalConflicts;
        return new NQueensEvaluation(
                missingRows == 0,
                fixedViolations == 0,
                columnConflicts == 0,
                diagonalConflicts == 0,
                missingRows,
                fixedViolations,
                columnConflicts,
                diagonalConflicts,
                conflicts,
                Map.of(
                        "size", problem.size(),
                        "fixedQueenCount", fixedQueenCount(problem)));
    }

    public static NQueensEvaluation evaluateTokens(NQueensProblem problem, int[] solutionTokens) {
        Objects.requireNonNull(problem, "problem must not be null");
        NQueensTokenDecodeResult decoded = NQueensTokenCodec.decodeSolution(problem.size(), solutionTokens);
        NQueensEvaluation evaluation = evaluate(problem, decoded.solution());
        int conflicts = evaluation.conflictCount()
                + decoded.invalidTokenCount()
                + decoded.multiQueenRowCount();
        return new NQueensEvaluation(
                evaluation.complete() && decoded.emptyRowCount() == 0,
                evaluation.respectsFixedQueens(),
                evaluation.uniqueColumns(),
                evaluation.diagonalSafe(),
                evaluation.missingRowCount(),
                evaluation.fixedViolationCount(),
                evaluation.columnConflictCount(),
                evaluation.diagonalConflictCount(),
                conflicts,
                Map.of(
                        "size", problem.size(),
                        "fixedQueenCount", fixedQueenCount(problem),
                        "invalidTokenCount", decoded.invalidTokenCount(),
                        "multiQueenRowCount", decoded.multiQueenRowCount(),
                        "queenTokenCount", decoded.queenTokenCount()));
    }

    public static DiscreteTokenEvaluation evaluateAsDiscrete(
            NQueensProblem problem,
            NQueensSolution solution) {
        Objects.requireNonNull(solution, "solution must not be null");
        return toDiscreteEvaluation(evaluate(problem, solution), solution.canonicalKey());
    }

    public static DiscreteTokenEvaluation evaluateTokensAsDiscrete(
            NQueensProblem problem,
            int[] solutionTokens) {
        Objects.requireNonNull(problem, "problem must not be null");
        NQueensTokenDecodeResult decoded = NQueensTokenCodec.decodeSolution(problem.size(), solutionTokens);
        return toDiscreteEvaluation(evaluateTokens(problem, solutionTokens), decoded.solution().canonicalKey());
    }

    public static NQueensCoverageReport coverage(NQueensProblem problem, List<NQueensSolution> candidates) {
        return coverage(problem, candidates, -1);
    }

    public static NQueensCoverageReport coverageTokens(NQueensProblem problem, List<int[]> candidateTokens) {
        return coverageTokens(problem, candidateTokens, -1);
    }

    public static NQueensCoverageReport coverageTokens(
            NQueensProblem problem,
            List<int[]> candidateTokens,
            int knownSolutionCount) {
        Objects.requireNonNull(problem, "problem must not be null");
        List<int[]> candidateList = List.copyOf(
                Objects.requireNonNull(candidateTokens, "candidateTokens must not be null"));
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }

        List<DiscreteTokenEvaluation> evaluations = new ArrayList<>(candidateList.size());
        for (int[] tokens : candidateList) {
            evaluations.add(evaluateTokensAsDiscrete(problem, tokens));
        }

        return toNQueensCoverageReport(DiscreteTokenCoverage.summarize(evaluations, knownSolutionCount));
    }

    public static NQueensCoverageReport coverageAgainstAllSolutions(
            NQueensProblem problem,
            List<NQueensSolution> candidates) {
        Objects.requireNonNull(problem, "problem must not be null");
        return coverage(problem, candidates, NQueensSolver.count(problem));
    }

    public static NQueensCoverageReport coverage(
            NQueensProblem problem,
            List<NQueensSolution> candidates,
            int knownSolutionCount) {
        Objects.requireNonNull(problem, "problem must not be null");
        List<NQueensSolution> candidateList = List.copyOf(
                Objects.requireNonNull(candidates, "candidates must not be null"));
        if (knownSolutionCount < -1) {
            throw new IllegalArgumentException("knownSolutionCount must be -1 or >= 0 but was " + knownSolutionCount);
        }

        List<DiscreteTokenEvaluation> evaluations = new ArrayList<>(candidateList.size());
        for (NQueensSolution candidate : candidateList) {
            evaluations.add(evaluateAsDiscrete(problem, candidate));
        }

        return toNQueensCoverageReport(DiscreteTokenCoverage.summarize(evaluations, knownSolutionCount));
    }

    private static DiscreteTokenEvaluation toDiscreteEvaluation(NQueensEvaluation evaluation, String canonicalKey) {
        return evaluation.valid()
                ? DiscreteTokenEvaluation.valid(canonicalKey, evaluation.metadata())
                : DiscreteTokenEvaluation.invalid(evaluation.conflictCount(), evaluation.metadata());
    }

    static NQueensCoverageReport toNQueensCoverageReport(DiscreteTokenCoverageReport report) {
        return new NQueensCoverageReport(
                report.candidateCount(),
                report.validCandidateCount(),
                report.uniqueValidCandidateCount(),
                report.duplicateValidCandidateCount(),
                report.knownSolutionCount(),
                report.validRate(),
                report.coverageRate());
    }

    private static int fixedQueenCount(NQueensProblem problem) {
        int count = 0;
        for (int row = 0; row < problem.size(); row++) {
            if (problem.hasFixedQueen(row)) {
                count++;
            }
        }
        return count;
    }
}
