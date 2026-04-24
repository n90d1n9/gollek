package tech.kayys.gollek.ml.pipeline;

import tech.kayys.gollek.ml.base.*;
import java.util.*;

/**
 * ColumnTransformer applies different transformers to different columns.
 * Essential for heterogeneous datasets with mixed data types.
 */
public class ColumnTransformer extends BaseTransformer {

    private final List<TransformerColumn> transformers;
    private final String remainder; // "drop", "passthrough", or transformer
    private List<String> featureNames;
    private int nFeaturesOut;

    public ColumnTransformer() {
        this.transformers = new ArrayList<>();
        this.remainder = "drop";
    }

    public ColumnTransformer addTransformer(String name, BaseTransformer transformer, int[] columns) {
        transformers.add(new TransformerColumn(name, transformer, columns));
        return this;
    }

    public ColumnTransformer addTransformer(String name, BaseTransformer transformer, String columns) {
        // For named column selection (when feature names are available)
        transformers.add(new TransformerColumn(name, transformer, columns));
        return this;
    }

    @Override
    public void fit(float[][] X) {
        int nFeatures = X[0].length;
        Set<Integer> usedColumns = new HashSet<>();

        // Fit each transformer on its columns
        for (TransformerColumn tc : transformers) {
            float[][] subset = extractColumns(X, tc.columns);
            tc.transformer.fit(subset);
            for (int col : tc.columns) {
                usedColumns.add(col);
            }
        }

        // Handle remainder columns
        if ("passthrough".equals(remainder)) {
            List<Integer> remainderCols = new ArrayList<>();
            for (int i = 0; i < nFeatures; i++) {
                if (!usedColumns.contains(i)) {
                    remainderCols.add(i);
                }
            }
            // Could add a passthrough transformer
        }

        // Calculate output feature count
        nFeaturesOut = 0;
        for (TransformerColumn tc : transformers) {
            nFeaturesOut += tc.transformer.getNFeaturesOut();
        }
        if ("passthrough".equals(remainder)) {
            nFeaturesOut += (nFeatures - usedColumns.size());
        }

        setFitted(true);
    }

    @Override
    public float[][] transform(float[][] X) {
        if (!isFitted()) {
            throw new IllegalStateException("ColumnTransformer must be fitted before transform");
        }

        List<float[]> allOutputs = new ArrayList<>();

        // Transform each column group
        for (TransformerColumn tc : transformers) {
            float[][] subset = extractColumns(X, tc.columns);
            float[][] transformed = tc.transformer.transform(subset);
            for (float[] row : transformed) {
                allOutputs.add(row);
            }
        }

        // Combine results
        int nSamples = X.length;
        float[][] result = new float[nSamples][nFeaturesOut];
        int colOffset = 0;

        for (TransformerColumn tc : transformers) {
            float[][] transformed = tc.transformer.transform(extractColumns(X, tc.columns));
            int nOut = transformed[0].length;
            for (int i = 0; i < nSamples; i++) {
                System.arraycopy(transformed[i], 0, result[i], colOffset, nOut);
            }
            colOffset += nOut;
        }

        return result;
    }

    private float[][] extractColumns(float[][] X, int[] columns) {
        int nSamples = X.length;
        float[][] subset = new float[nSamples][columns.length];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < columns.length; j++) {
                subset[i][j] = X[i][columns[j]];
            }
        }

        return subset;
    }

    @Override
    public List<String> getFeatureNames(List<String> inputFeatures) {
        List<String> names = new ArrayList<>();
        for (TransformerColumn tc : transformers) {
            List<String> tcNames = tc.transformer.getFeatureNames(
                    inputFeatures != null
                            ? Arrays.asList(
                                    Arrays.stream(tc.columns).mapToObj(inputFeatures::get).toArray(String[]::new))
                            : null);
            names.addAll(tcNames);
        }
        return names;
    }

    @Override
    public int getNFeaturesOut() {
        return nFeaturesOut;
    }

    private static class TransformerColumn {
        final String name;
        final BaseTransformer transformer;
        final int[] columns;
        final String columnNames; // For named columns

        TransformerColumn(String name, BaseTransformer transformer, int[] columns) {
            this.name = name;
            this.transformer = transformer;
            this.columns = columns;
            this.columnNames = null;
        }

        TransformerColumn(String name, BaseTransformer transformer, String columnNames) {
            this.name = name;
            this.transformer = transformer;
            this.columnNames = columnNames;
            this.columns = null;
        }
    }
}