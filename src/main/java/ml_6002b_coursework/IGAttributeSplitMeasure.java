package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

/**
 * CMP-6002B Machine Learning Classification with Decision Trees
 *
 * Provides an implementation of AttributeSplitMeasure using
 * Information Gain.
 *
 * Ref: Id3.java, Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 * @author Alex Middlemiss, 100219171, exb17gxu
 * @version 1.0, 21/03/2021
 */

public class IGAttributeSplitMeasure implements AttributeSplitMeasure {

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);

        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double)splitData[j].numInstances() /
                        (double)data.numInstances()) *
                        computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    /**
     * Computes the entropy of a dataset.
     *
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data's class distribution
     */
    private double computeEntropy(Instances data) {

        double [] classCounts = new double[data.numClasses()];
        double entropy = 0;
        Enumeration instEnum = data.enumerateInstances();

        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int)inst.classValue()]++;
        }

        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= data.numInstances();

        return entropy + Utils.log2(data.numInstances());
    }
}
