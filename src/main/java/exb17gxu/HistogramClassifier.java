package exb17gxu;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class HistogramClassifier implements Classifier {

    @Override
    public Capabilities getCapabilities() {

        return null;
    }

    @Override
    public void buildClassifier (Instances data) {

    }

    @Override
    public double classifyInstance (Instance instance) {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        return new double[]{0};
    }
}
