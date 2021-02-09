package exb17gxu;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;

public class Lab1 {

    static String rootPath = "/Users/Alex/OneDrive - University of East Anglia/Year 3/Machine Learning/Labs/Labs1/";

    public static Instances loadData (String fileName) {
        FileReader reader = null;
        try {
            reader = new FileReader(rootPath + fileName);
            return new Instances(reader);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main (String[] args) {
        Instances arsenalTrain = loadData("Arsenal_TRAIN.arff");
        Instances arsenalTest = loadData("Arsenal_TEST.arff");

        System.out.println("arsenalTrain numInstances = " + arsenalTrain.numInstances());
        System.out.println("arsenalTest numAttributes = " + arsenalTest.numAttributes());
        int numWins = 0;
        for (Instance instance : arsenalTrain) {
            if (instance.value(arsenalTrain.attribute(3)) == 2.0) {
                numWins++;
            }
        }
        System.out.println("arsenalTrain number wins = " + numWins);
        double[] fifth = arsenalTest.instance(4).toDoubleArray();
        for (double x : fifth) {
            System.out.println("arsenalTest 5th instances value = " + x);
        }
        for (Instance instance : arsenalTrain) {
            System.out.println("arsenalTrain instances = " + instance.toString());
        }

        arsenalTrain.deleteAttributeAt(2);
        for (Instance instance : arsenalTrain) {
            System.out.println("arsenalTrain instances = " + instance.toString());
        }

        arsenalTrain = loadData("Arsenal_TRAIN.arff");
        arsenalTrain.setClassIndex(arsenalTrain.numAttributes() - 1);
        arsenalTest.setClassIndex(arsenalTest.numAttributes() - 1);

        NaiveBayes nB = new NaiveBayes();
        try {
            nB.buildClassifier(arsenalTrain);
        } catch (Exception e) {
            e.printStackTrace();
        }
        int correct = 0;
        for (int i = 0; i < arsenalTest.numInstances(); i++) {
            try {
                double insClass = nB.classifyInstance(arsenalTest.instance(i));
                if (insClass == arsenalTest.instance(i).classValue()) {
                    correct++;
                }
                System.out.println("class = " + insClass);
                double[] insProb = nB.distributionForInstance(arsenalTest.instance(i));
                for (double d : insProb) {
                    System.out.print(d + ", ");
                }
                System.out.println();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println(correct + " instances predicted correctly");
        double acc = correct / (double) arsenalTest.numInstances();
        System.out.println("Accuracy = " + acc);

        IBk iBk = new IBk();
        try {
            iBk.buildClassifier(arsenalTrain);
        } catch (Exception e) {
            e.printStackTrace();
        }
        int correctiBk = 0;
        for (int i = 0; i < arsenalTest.numInstances(); i++) {
            try {
                double insClass = iBk.classifyInstance(arsenalTest.instance(i));
                if (insClass == arsenalTest.instance(i).classValue()) {
                    correctiBk++;
                }
                System.out.println("class = " + insClass);
                double[] insProb = iBk.distributionForInstance(arsenalTest.instance(i));
                for (double d : insProb) {
                    System.out.print(d + ", ");
                }
                System.out.println();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println(correctiBk + " instances predicted correctly");
        double acciBk = correctiBk / (double) arsenalTest.numInstances();
        System.out.println("Accuracy = " + acciBk);
    }
}
