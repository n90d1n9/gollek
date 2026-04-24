package tech.kayys.gollek.ml.feature;

/**
 * One-Hot Encoder for categorical features.
 */
public class OneHotEncoder extends BaseTransformer {
    private final boolean sparseOutput;
    private final int handleUnknown; // 0=error, 1=ignore
    private Map<Integer, Map<Object, Integer>> categoryMaps;
    private int[] nValues;

    public OneHotEncoder() {
        this(false, 0);
    }

    public OneHotEncoder(boolean sparseOutput, int handleUnknown) {
        this.sparseOutput = sparseOutput;
        this.handleUnknown = handleUnknown;
    }

    @Override
    public void fit(Object[][] X, int[] featureIndices) {
        categoryMaps = new HashMap<>();
        nValues = new int[featureIndices.length];

        for (int idx = 0; idx < featureIndices.length; idx++) {
            int col = featureIndices[idx];
            Map<Object, Integer> catMap = new LinkedHashMap<>();

            for (Object[] row : X) {
                Object value = row[col];
                if (!catMap.containsKey(value)) {
                    catMap.put(value, catMap.size());
                }
            }

            categoryMaps.put(col, catMap);
            nValues[idx] = catMap.size();
        }
    }

    public float[][] transform(Object[][] X) {
        // Calculate output dimensions
        int totalFeatures = 0;
        for (int n : nValues)
            totalFeatures += n;

        int nSamples = X.length;
        float[][] transformed = new float[nSamples][totalFeatures];

        int offset = 0;
        for (Map.Entry<Integer, Map<Object, Integer>> entry : categoryMaps.entrySet()) {
            int col = entry.getKey();
            Map<Object, Integer> catMap = entry.getValue();

            for (int i = 0; i < nSamples; i++) {
                Object value = X[i][col];
                Integer catIdx = catMap.get(value);

                if (catIdx != null) {
                    transformed[i][offset + catIdx] = 1.0f;
                } else if (handleUnknown == 0) {
                    throw new IllegalArgumentException("Unknown category: " + value);
                }
            }

            offset += catMap.size();
        }

        return transformed;
    }

    public List<String> getFeatureNames(List<String> inputFeatures) {
        List<String> featureNames = new ArrayList<>();

        for (Map.Entry<Integer, Map<Object, Integer>> entry : categoryMaps.entrySet()) {
            int col = entry.getKey();
            String inputName = inputFeatures.get(col);
            Map<Object, Integer> catMap = entry.getValue();

            for (Object category : catMap.keySet()) {
                featureNames.add(inputName + "_" + category);
            }
        }

        return featureNames;
    }
}