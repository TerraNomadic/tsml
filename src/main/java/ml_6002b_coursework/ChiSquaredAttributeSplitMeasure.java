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
 * Chi Squared Statistic.
 *
 * @author Alex Middlemiss, 100219171, exb17gxu
 * @version 1.0, 21/03/2021
 */

public class ChiSquaredAttributeSplitMeasure implements AttributeSplitMeasure {

    /**
     * Computes Chi Squared statistic for an attribute.
     *
     * @param data the data for which chi is to be computed
     * @param att  the attribute
     * @param chiYates toggle for the Yates correction to chi
     * @return the chi for the given attribute and data
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att,
                                          boolean chiYates) {

        double chi = 0.0;
        Instances[] splitData = splitData(data, att);
        double[] dataClassCount = new double[data.numClasses()];
        double[][] splitClassCounts = new double[splitData.length][data.numClasses()];

        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            dataClassCount[(int)inst.classValue()]++;
        }

        for (int i = 0; i < splitData.length; i++) {
            Enumeration instEnum2 = splitData[i].enumerateInstances();
            while (instEnum2.hasMoreElements()) {
                Instance inst = (Instance) instEnum2.nextElement();
                splitClassCounts[i][(int) inst.classValue()]++;
            }
        }

        for (int i = 0; i < att.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                for (int j = 0; j < splitClassCounts[i].length; j++) {
                    double exp = (splitData[i].numInstances() *
                            (dataClassCount[j] / data.numInstances()));
                    if (chiYates) {
                        chi += Math.pow((splitClassCounts[i][j] - exp - 0.5), 2) / exp;
                    } else {
                        chi += Math.pow((splitClassCounts[i][j] - exp), 2) / exp;
                    }
                }
            }
        }

        return chi;
    }

    public static void main (String[] args) throws Exception {
        String basePath = "src/main/java/ml_6002b_coursework/test_data/";
        String dataset = "Meningitis";
        Vector<String> headacheValues = new Vector<>(2);
        headacheValues.add("0"); headacheValues.add("1");
        Attribute headache = new Attribute("headache", headacheValues);

        ChiSquaredAttributeSplitMeasure chi = new ChiSquaredAttributeSplitMeasure();

        Instances meningitis = DatasetLoading.loadDataThrowable(basePath + dataset + "_TRAIN.arff");
        //Instances meningitis = loadClassificationData(basePath + dataset + "_TRAIN.arff");

        System.out.println("chi Headache = " + chi.computeAttributeQuality(meningitis, headache, false));
        System.out.println("chi yates Headache = " + chi.computeAttributeQuality(meningitis, headache, true));
    }
}