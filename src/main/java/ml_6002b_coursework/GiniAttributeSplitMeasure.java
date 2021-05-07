package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.Vector;

/**
 * CMP-6002B Machine Learning Classification with Decision Trees
 *
 * Provides an implementation of AttributeSplitMeasure using
 * Gini Index.
 *
 * @author Alex Middlemiss, 100219171, exb17gxu
 * @version 1.0, 21/03/2021
 */

public class GiniAttributeSplitMeasure implements AttributeSplitMeasure {

    /**
     * Computes gini index for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the gini index for the given attribute and data
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) {

        double gini = computeImpurity(data);
        Instances[] splitData = splitData(data, att);

        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                gini -= ((double)splitData[j].numInstances() /
                        (double)data.numInstances()) *
                        computeImpurity(splitData[j]);
            }
        }
        return gini;
    }

    /**
     * Computes the impurity of a dataset.
     *
     * @param data the data for which impurity is to be computed
     * @return the impurity of the data's class distribution
     */
    private double computeImpurity(Instances data) {

        double [] classCounts = new double[data.numClasses()];
        double impurity = 1.0;
        Enumeration instEnum = data.enumerateInstances();

        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int)inst.classValue()]++;
        }

        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                impurity -= Math.pow(classCounts[j] / data.numInstances(), 2);
            }
        }

        return impurity;
    }

    public static void main (String[] args) throws Exception {
        String basePath = "src/main/java/ml_6002b_coursework/test_data/";
        String dataset = "Meningitis";
        Vector<String> headacheValues = new Vector<>(2);
        headacheValues.add("0"); headacheValues.add("1");
        Attribute headache = new Attribute("headache", headacheValues);

        GiniAttributeSplitMeasure gini = new GiniAttributeSplitMeasure();

        Instances meningitis = DatasetLoading.loadDataThrowable(basePath + dataset + "_TRAIN.arff");
        //Instances meningitis = loadClassificationData(basePath + dataset + "_TRAIN.arff");

        System.out.println("gini Headache = " + gini.computeAttributeQuality(meningitis, headache));
    }
}
