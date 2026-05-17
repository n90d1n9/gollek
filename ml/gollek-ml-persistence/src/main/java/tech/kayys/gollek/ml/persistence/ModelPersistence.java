package tech.kayys.gollek.ml.persistence;

import tech.kayys.gollek.ml.base.BaseEstimator;
import tech.kayys.gollek.ml.ensemble.RandomForestClassifier;
import tech.kayys.gollek.ml.linear_model.LinearModel;
import tech.kayys.gollek.ml.tree.DecisionTreeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Saves and loads ML estimators using a simple Gollek-native serialized state.
 */
public final class ModelPersistence {

    private ModelPersistence() {
    }

    public static void save(BaseEstimator model, String path) throws IOException {
        save(model, new File(path));
    }

    public static void save(BaseEstimator model, File file) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new GZIPOutputStream(new FileOutputStream(file)))) {
            Map<String, Object> state = extractState(model);
            oos.writeObject(state);
        }
    }

    public static <T extends BaseEstimator> T load(String path) throws IOException, ClassNotFoundException {
        return load(new File(path));
    }

    @SuppressWarnings("unchecked")
    public static <T extends BaseEstimator> T load(File file) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(file)))) {
            Map<String, Object> state = (Map<String, Object>) ois.readObject();
            String className = (String) state.get("_class");
            Class<?> modelClass = Class.forName(className);
            T model = (T) modelClass.getDeclaredConstructor().newInstance();
            restoreState(model, state);
            return model;
        } catch (IOException | ClassNotFoundException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to load model", e);
        }
    }

    public static String toPMML(BaseEstimator model, String modelName,
            List<String> featureNames, List<String> targetNames) {
        StringBuilder pmml = new StringBuilder();

        pmml.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        pmml.append("<PMML version=\"4.4\" xmlns=\"http://www.dmg.org/PMML-4_4\">\n");
        pmml.append("  <Header>\n");
        pmml.append("    <Application name=\"Gollek ML\"/>\n");
        pmml.append("    <Timestamp>").append(new Date()).append("</Timestamp>\n");
        pmml.append("  </Header>\n");
        pmml.append("  <DataDictionary>\n");

        for (String feature : featureNames) {
            pmml.append("    <DataField name=\"").append(feature)
                    .append("\" optype=\"continuous\" dataType=\"double\"/>\n");
        }
        for (String target : targetNames) {
            pmml.append("    <DataField name=\"").append(target)
                    .append("\" optype=\"categorical\" dataType=\"string\"/>\n");
        }
        pmml.append("  </DataDictionary>\n");

        if (model instanceof DecisionTreeClassifier tree) {
            pmml.append(exportTreeToPMML(tree, featureNames));
        } else if (model instanceof RandomForestClassifier forest) {
            pmml.append(exportForestToPMML(forest, featureNames));
        } else if (model instanceof LinearModel linear) {
            pmml.append(exportLinearToPMML(linear, featureNames, targetNames));
        }

        pmml.append("</PMML>");
        return pmml.toString();
    }

    private static String exportTreeToPMML(DecisionTreeClassifier tree, List<String> features) {
        return "    <TreeModel modelName=\"DecisionTree\" functionName=\"classification\">\n" +
                "      <Node id=\"1\" score=\"0\">\n" +
                "        <True/>\n" +
                "      </Node>\n" +
                "    </TreeModel>\n";
    }

    private static String exportForestToPMML(RandomForestClassifier forest, List<String> features) {
        return "    <MiningModel modelName=\"RandomForest\" functionName=\"classification\">\n" +
                "      <MiningSchema>\n" +
                "        <MiningField name=\"predicted\"/>\n" +
                "      </MiningSchema>\n" +
                "      <Segmentation multipleModelMethod=\"majorityVote\">\n" +
                "        <!-- Segment for each tree -->\n" +
                "      </Segmentation>\n" +
                "    </MiningModel>\n";
    }

    private static String exportLinearToPMML(LinearModel model, List<String> features, List<String> targets) {
        double[] coef = model.getCoefficients();
        double intercept = model.getIntercept();

        StringBuilder pmml = new StringBuilder();
        pmml.append("    <RegressionModel modelName=\"LinearRegression\" functionName=\"regression\">\n");
        pmml.append("      <MiningSchema>\n");
        pmml.append("        <MiningField name=\"predicted\"/>\n");
        pmml.append("      </MiningSchema>\n");
        pmml.append("      <RegressionTable intercept=\"").append(intercept).append("\">\n");

        for (int i = 0; i < features.size() && i < coef.length; i++) {
            pmml.append("        <NumericPredictor name=\"").append(features.get(i))
                    .append("\" coefficient=\"").append(coef[i]).append("\"/>\n");
        }

        pmml.append("      </RegressionTable>\n");
        pmml.append("    </RegressionModel>\n");
        return pmml.toString();
    }

    private static Map<String, Object> extractState(BaseEstimator model) {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("_class", model.getClass().getName());

        try {
            for (java.lang.reflect.Field field : model.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                Object value = field.get(model);
                if (value != null && isSerializable(value)) {
                    state.put(field.getName(), value);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to extract model state", e);
        }

        return state;
    }

    private static void restoreState(BaseEstimator model, Map<String, Object> state) {
        try {
            for (java.lang.reflect.Field field : model.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                if (state.containsKey(field.getName())) {
                    field.set(model, state.get(field.getName()));
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to restore model state", e);
        }
    }

    private static boolean isSerializable(Object obj) {
        return obj instanceof Serializable ||
                obj instanceof Number ||
                obj instanceof String ||
                obj instanceof Boolean ||
                obj instanceof Date ||
                (obj instanceof Object[] values && values.length > 0 &&
                        isSerializable(values[0]));
    }
}
