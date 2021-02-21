package exb17gxu;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class MajorityClassifier extends AbstractClassifier {

    private int maxClassValue;
    private int[] classValues;
    private int numClasses;

    @Override
    public void buildClassifier (Instances data) {
        int maxClassCount;
        numClasses = data.numClasses();
        classValues = new int[numClasses];

        for (Instance ins : data) {
            classValues[(int) ins.classValue()]++;
        }
        maxClassCount = classValues[0];
        maxClassValue = 0;
        for (int i = 0; i < classValues.length; i++) {
            if (classValues[i] > maxClassCount) {
                maxClassValue = i;
            }
        }
    }

    @Override
    public double classifyInstance (Instance instance) {
        return maxClassValue;
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] dist = new double[numClasses];
        dist[maxClassValue] = 1.0;
        return dist;
    }
}
