package exb17gxu;

import experiments.data.DatasetLoading;
import weka.attributeSelection.ScatterSearchV1;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.HashSet;

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
        for (Instance ins : data) {
            dist[(int) ins.classValue()]++;
        }
        for (int i = 0; i < dist.length; i++) {
            dist[i] = dist[i] / data.numInstances();
        }
        return dist;
    }

    public static int[][] confusionMatrix(int[] predicted, int[] actual) {
        if (predicted.length != actual.length) {
            System.err.println("Error: predicted and actual not same length.");
            return null;
        }
        int numClasses = 0;
        HashSet<Integer> hs = new HashSet<Integer>();
        for (int i : actual) {
            hs.add(i);
        }
        numClasses = hs.size();

        int[][] confusionMatrix = new int[numClasses][numClasses];
        for (int i = 0; i < actual.length; i++) {
            confusionMatrix[predicted[i]][actual[i]]++;
        }
        return confusionMatrix;
    }

    public static void printConfMatrix(int[][] cM) {
        System.out.println("    A       ");
        System.out.println("P   0, 1");
        System.out.println("0   " + cM[0][0] + ", " + cM[0][1]);
        System.out.println("1   " + cM[1][0] + ", " + cM[1][1]);
    }

    public static int[] classifyInstances(Classifier c, Instances test) {
        int[] pred = new int[test.numInstances()];
        for (int i = 0; i < test.numInstances(); i++) {
            try {
                pred[i] = (int) c.classifyInstance(test.instance(i));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return pred;
    }

    public static int[] getClassValues(Instances data) {
        int[] classValues = new int[data.numInstances()];
        for (int i = 0; i < data.numInstances(); i++) {
            classValues[i] = (int) data.instance(i).classValue();
        }
        return classValues;
    }

    public static void main (String[] args) {
        String basePath = "src/main/java/experiments/data/tsc/";
        String dataset = "ItalyPowerDemand";

        Instances train = loadClassificationData(basePath + dataset + "/" + dataset + "_TRAIN.arff");
        Instances test = loadClassificationData(basePath + dataset + "/" + dataset + "_TEST.arff");

        System.out.println("train numInstances = " + train.numInstances());
        System.out.println("test numInstances = " + test.numInstances());
        System.out.println("train numAttributes = " + train.numAttributes());
        System.out.println("test numAttributes = " + test.numAttributes());

        MajorityClassifier mC = new MajorityClassifier();
        try {
            mC.buildClassifier(train);
        } catch (Exception e) {
            e.printStackTrace();
        }

        double acc = accuracy(mC, test);
        System.out.println("Accuracy = " + acc);

        int[] actual = getClassValues(test);
        int[] predicted = classifyInstances(mC, test);
        System.out.println("Actual length = " + actual.length);
        System.out.println("Predicted length = " + predicted.length);

        int[][] cM = confusionMatrix(predicted, actual);
        System.out.println("Confusion = ");
        printConfMatrix(cM);

        ZeroR zC = new ZeroR();
        try {
            zC.buildClassifier(train);
        } catch (Exception e) {
            e.printStackTrace();
        }

        double accZC = accuracy(zC, test);
        System.out.println("Accuracy = " + accZC);

        int[] actualZC = getClassValues(test);
        int[] predictedZC = classifyInstances(zC, test);
        System.out.println("Actual length = " + actualZC.length);
        System.out.println("Predicted length = " + predictedZC.length);

        int[][] zCM = confusionMatrix(predictedZC, actualZC);
        System.out.println("Confusion = ");
        printConfMatrix(zCM);
    }
}
