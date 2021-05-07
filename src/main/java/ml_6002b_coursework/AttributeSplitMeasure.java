package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 *
 */
public interface AttributeSplitMeasure {

    default double computeAttributeQuality(Instances data, Attribute att) {
        throw new UnsupportedOperationException();
    }

    default double computeAttributeQuality(Instances data, Attribute att,
                                   boolean chiYates) {
        throw new UnsupportedOperationException();
    }

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
     default Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            //splitData[(int)inst.value(att)].add(inst);
            if (inst.value(0) == 0.0) {
                splitData[0].add(inst);
            } else {
                splitData[1].add(inst); //TODO: Remove!!!!
            }

        }
         for (Instances splitDatum : splitData) {
             splitDatum.compactify();
         }
        return splitData;
    }

}
