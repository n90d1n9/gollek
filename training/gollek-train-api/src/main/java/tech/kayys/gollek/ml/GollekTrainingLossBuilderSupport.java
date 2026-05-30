package tech.kayys.gollek.ml;

import static tech.kayys.gollek.ml.GollekTrainingOptionValidators.*;

import tech.kayys.gollek.train.data.DataLoader;

/**
 * Fluent loss-parameter configuration shared by training option builders.
 */
public abstract class GollekTrainingLossBuilderSupport<B extends GollekTrainingLossBuilderSupport<B>>
        extends GollekTrainingSchedulerBuilderSupport<B> {
    protected float[] crossEntropyClassWeights;
    protected Float focalGamma;
    protected Float focalAlpha;
    protected float[] focalClassWeights;
    protected Float causalLanguageModelingIgnoreIndex;
    protected float[] bcePositiveWeights;
    protected float[] pinballQuantiles;
    protected Float intervalAlpha;
    protected Float intervalCrossingPenalty;
    protected Float tweediePower;
    protected Boolean tweedieLogInput;
    protected Float tweedieEps;
    protected Boolean negativeBinomialLogInput;
    protected Boolean negativeBinomialIncludeConstant;
    protected Float negativeBinomialEps;
    protected Boolean zeroInflatedPoissonLogRateInput;
    protected Boolean zeroInflatedPoissonIncludeConstant;
    protected Float zeroInflatedPoissonEps;
    protected Boolean zeroInflatedNegativeBinomialLogInput;
    protected Boolean zeroInflatedNegativeBinomialIncludeConstant;
    protected Float zeroInflatedNegativeBinomialEps;

    public B crossEntropyClassWeights(float... classWeights) {
        this.crossEntropyClassWeights = normalizeCrossEntropyClassWeights(classWeights);
        return self();
    }

    public B classWeights(int... labels) {
        this.crossEntropyClassWeights = DataLoader.classWeights(labels);
        return self();
    }

    public B classWeightsFor(int numClasses, int... labels) {
        this.crossEntropyClassWeights = DataLoader.classWeightsFor(numClasses, labels);
        return self();
    }

    public B focalGamma(float gamma) {
        this.focalGamma = normalizeFocalGamma(gamma);
        return self();
    }

    public B focalAlpha(float alpha) {
        this.focalAlpha = normalizeFocalAlpha(alpha);
        return self();
    }

    public B focal(float gamma, float alpha) {
        return focalGamma(gamma).focalAlpha(alpha);
    }

    public B focalClassWeights(float... classWeights) {
        this.focalClassWeights = normalizeFocalClassWeights(classWeights);
        return self();
    }

    public B focalClassWeights(int... labels) {
        this.focalClassWeights = DataLoader.classWeights(labels);
        return self();
    }

    public B focalClassWeightsFor(int numClasses, int... labels) {
        this.focalClassWeights = DataLoader.classWeightsFor(numClasses, labels);
        return self();
    }

    public B causalLanguageModelingIgnoreIndex(float ignoreIndex) {
        this.causalLanguageModelingIgnoreIndex = normalizeCausalLanguageModelingIgnoreIndex(ignoreIndex);
        return self();
    }

    public B nextTokenIgnoreIndex(float ignoreIndex) {
        return causalLanguageModelingIgnoreIndex(ignoreIndex);
    }

    public B bcePositiveWeight(float positiveWeight) {
        this.bcePositiveWeights = normalizeBcePositiveWeights(new float[] { positiveWeight });
        return self();
    }

    public B bcePositiveWeights(float... positiveWeights) {
        this.bcePositiveWeights = normalizeBcePositiveWeights(positiveWeights);
        return self();
    }

    public B pinballQuantile(double quantile) {
        this.pinballQuantiles = normalizePinballQuantiles(new float[] { (float) quantile });
        return self();
    }

    public B pinballQuantiles(double... quantiles) {
        this.pinballQuantiles = normalizePinballQuantiles(toFloatArray(quantiles));
        return self();
    }

    public B predictionIntervalQuantiles(double lowerQuantile, double upperQuantile) {
        if (lowerQuantile >= upperQuantile) {
            throw new IllegalArgumentException(
                    "lowerQuantile must be less than upperQuantile, got: "
                            + lowerQuantile + " >= " + upperQuantile);
        }
        return pinballQuantiles(lowerQuantile, upperQuantile);
    }

    public B intervalScoreAlpha(double alpha) {
        this.intervalAlpha = normalizeIntervalAlpha((float) alpha);
        return self();
    }

    public B predictionIntervalAlpha(double alpha) {
        return intervalScoreAlpha(alpha);
    }

    public B intervalCrossingPenalty(double crossingPenalty) {
        this.intervalCrossingPenalty = normalizeIntervalCrossingPenalty((float) crossingPenalty);
        return self();
    }

    public B tweediePower(double power) {
        this.tweediePower = normalizeTweediePower((float) power);
        return self();
    }

    public B compoundPoissonGammaPower(double power) {
        return tweediePower(power);
    }

    public B tweedieLogInput(boolean logInput) {
        this.tweedieLogInput = logInput;
        return self();
    }

    public B tweedieRawMeanInput() {
        return tweedieLogInput(false);
    }

    public B tweedieEps(double eps) {
        this.tweedieEps = normalizePositiveFloat((float) eps, "tweedieEps");
        return self();
    }

    public B negativeBinomialLogInput(boolean logInput) {
        this.negativeBinomialLogInput = logInput;
        return self();
    }

    public B negativeBinomialRawInput() {
        return negativeBinomialLogInput(false);
    }

    public B negativeBinomialIncludeConstant(boolean includeConstant) {
        this.negativeBinomialIncludeConstant = includeConstant;
        return self();
    }

    public B negativeBinomialFullNll() {
        return negativeBinomialIncludeConstant(true);
    }

    public B negativeBinomialEps(double eps) {
        this.negativeBinomialEps = normalizePositiveFloat((float) eps, "negativeBinomialEps");
        return self();
    }

    public B zeroInflatedPoissonLogRateInput(boolean logRateInput) {
        this.zeroInflatedPoissonLogRateInput = logRateInput;
        return self();
    }

    public B zeroInflatedPoissonRawRateInput() {
        return zeroInflatedPoissonLogRateInput(false);
    }

    public B zeroInflatedPoissonIncludeConstant(boolean includeConstant) {
        this.zeroInflatedPoissonIncludeConstant = includeConstant;
        return self();
    }

    public B zeroInflatedPoissonFullNll() {
        return zeroInflatedPoissonIncludeConstant(true);
    }

    public B zeroInflatedPoissonEps(double eps) {
        this.zeroInflatedPoissonEps = normalizePositiveFloat((float) eps, "zeroInflatedPoissonEps");
        return self();
    }

    public B zeroInflatedNegativeBinomialLogInput(boolean logInput) {
        this.zeroInflatedNegativeBinomialLogInput = logInput;
        return self();
    }

    public B zeroInflatedNegativeBinomialRawInput() {
        return zeroInflatedNegativeBinomialLogInput(false);
    }

    public B zeroInflatedNegativeBinomialIncludeConstant(boolean includeConstant) {
        this.zeroInflatedNegativeBinomialIncludeConstant = includeConstant;
        return self();
    }

    public B zeroInflatedNegativeBinomialFullNll() {
        return zeroInflatedNegativeBinomialIncludeConstant(true);
    }

    public B zeroInflatedNegativeBinomialEps(double eps) {
        this.zeroInflatedNegativeBinomialEps = normalizePositiveFloat(
                (float) eps,
                "zeroInflatedNegativeBinomialEps");
        return self();
    }

    public B binaryPositiveWeight(int... labels) {
        return bcePositiveWeight(DataLoader.binaryPositiveWeight(labels));
    }

    public B binaryPositiveWeight(boolean... labels) {
        return bcePositiveWeight(DataLoader.binaryPositiveWeight(labels));
    }

    public B binaryPositiveWeight(float... labels) {
        return bcePositiveWeight(DataLoader.binaryPositiveWeight(labels));
    }

    public B multiLabelPositiveWeights(int[][] labels) {
        return bcePositiveWeights(DataLoader.multiLabelPositiveWeights(labels));
    }

    public B multiLabelPositiveWeights(boolean[][] labels) {
        return bcePositiveWeights(DataLoader.multiLabelPositiveWeights(labels));
    }

    public B multiLabelPositiveWeights(float[][] labels) {
        return bcePositiveWeights(DataLoader.multiLabelPositiveWeights(labels));
    }
}
