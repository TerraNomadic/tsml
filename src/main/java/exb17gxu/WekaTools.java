package exb17gxu;

import org.checkerframework.checker.units.qual.A;
import weka.classifiers.Classifier;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;

public class WekaTools {

    public static double accuracy(Classifier c, Instances test) {
        int correct = 0;
        for (Instance instance : test) {
            try {
                if (instance.classValue() == c.classifyInstance(instance)) {
                    correct++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return correct / (double) test.numInstances();
    }

    public static Instances loadClassificationData (String fullPath) {
        try {
            FileReader reader = new FileReader(fullPath);
            Instances instances = new Instances(reader);
            instances.setClassIndex(instances.numAttributes() - 1);
            return instances;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static Instances[] splitData(Instances all, double proportion) {
        Instances[] split = new Instances[2];
        split[0] = new Instances(all);
        split[1] = new Instances(all, 0);
        Debug.Random rand = new Debug.Random();
        all.randomize(rand);
        for (int i = 0; i < proportion * all.numInstances(); i++) {
            split[1].add(split[0].remove(i));
        }
        return split;
    }

    public static double[] classDistribution(Instances data) {
        double[] dist = new double[data.numClasses()];

        /*HashMap<Double, Integer> values = new HashMap<>();
        for (Instance ins : data) {
            if (!values.containsKey(ins.classValue())) {
                values.put(ins.classValue(), 1);
            } else {
                values.replace(ins.classValue(),values.get(ins.classValue())+1);
            }
        }*/

        for (Instance ins : data) {
            dist[(int) ins.classValue()]++;
        }
        for (int i = 0; i < dist.length; i++) {
            dist[i] = dist[i] / data.numInstances();
        }
        return dist;
    }
}
