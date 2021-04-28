package ml_6002b_coursework;

import java.util.Arrays;

/**
 * CMP-6002B Machine Learning Classification with Decision Trees
 *
 * Provides various measures to assess the quality of a split at
 * a node in a decision tree.
 *
 * @author Alex Middlemiss, 100219171, exb17gxu
 * @version 1.0, 21/03/2021
 */

public class AttributeMeasures {

    /**
     * Calculates the entropy for the class counts according to
     * the formula: H=-sum[i=1->c](P(i)log(P(i)))
     *
     * @param classCounts an array of the values for the node being assessed
     * @return the entropy of the node as a double
     */
    public static double calcEntropy(int[] classCounts) {

        double entropy = 0;
        int classSum = Arrays.stream(classCounts).sum();

        for (int count : classCounts) {
            entropy += (count / (float)classSum) * logBase2(count / (float)classSum);
        }

        return -entropy;
    }

    /**
     * Measures the information gain for the contingency table
     * according to the formula: Gain(X,A)=H(X)-sum[Y⍷V]((|Y|/|X|)*H(Y))
     *
     * @param att_table the contingency table for the attribute
     * @return the information gain as a double
     */
    public static double measureInformationGain(int[][] att_table) {

        int[] colSums = new int[att_table[0].length];
        int[] rowSums = new int[att_table.length];
        double[] attValueEntropies = new double[att_table.length];
        boolean first = true;

        for (int j = 0; j < att_table[0].length; j++) {
            int colSum = 0;
            for (int i = 0; i < att_table.length; i++) {
                colSum += att_table[i][j];
                if (first) {
                    attValueEntropies[i] = calcEntropy(att_table[i]);
                    rowSums[i] = Arrays.stream(att_table[i]).sum();
                }
            }
            colSums[j] = colSum;
            first = false;
        }

        double gain = calcEntropy(colSums);
        int colSumsSum = Arrays.stream(colSums).sum();
        for (int i = 0; i < attValueEntropies.length; i++) {
            gain -= (rowSums[i] / (float)colSumsSum) * attValueEntropies[i];
        }

        return gain;
    }

    /**
     * Calculates the gini impurity for the class counts according to
     * the formula: H=1-sum[i=1->c](P(i)^2)
     *
     * @param classCounts an array of the values for the node being assessed
     * @return the impurity of the node as a double
     */
    public static double calcImpurity(int[] classCounts) {

        double impurity = 0;
        int classSum = Arrays.stream(classCounts).sum();

        for (int count : classCounts) {
            impurity += Math.pow((count / (float)classSum),2);
        }

        return 1-impurity;
    }

    /**
     * Measures the gini index for the contingency table
     * according to the formula: Gini(X,A)=I(X)-sum[[Y⍷V]]((|Y|/|X|)*I(Y))
     *
     * @param att_table the contingency table for the attribute
     * @return the gini index as a double
     */
    public static double measureGini(int[][] att_table) {

        int[] colSums = new int[att_table[0].length];
        int[] rowSums = new int[att_table.length];
        double[] attValueImps = new double[att_table.length];
        boolean first = true;

        for (int j = 0; j < att_table[0].length; j++) {
            int colSum = 0;
            for (int i = 0; i < att_table.length; i++) {
                colSum += att_table[i][j];
                if (first) {
                    attValueImps[i] = calcImpurity(att_table[i]);
                    rowSums[i] = Arrays.stream(att_table[i]).sum();
                }
            }
            colSums[j] = colSum;
            first = false;
        }

        double gini = calcImpurity(colSums);
        int colSumsSum = Arrays.stream(colSums).sum();
        for (int i = 0; i < attValueImps.length; i++) {
            gini -= (rowSums[i] / (float)colSumsSum) * attValueImps[i];
        }

        return gini;
    }

    /**
     * Measures the chi-squared statistic for the contingency table
     * according to the formula:
     * x^2=sum[i=1->r](sum[j=1->c]((oij-eij)^2/eij)))
     *
     * @param att_table the contingency table for the attribute
     * @return the chi-squared index as a double
     */
    public static double measureChiSquared(int[][] att_table) {

        int[] colSums = new int[att_table[0].length];
        int[] rowSums = new int[att_table.length];
        double[][] exp_values = new double[att_table.length][att_table[0].length];
        boolean first = true;

        for (int j = 0; j < att_table[0].length; j++) {
            int colSum = 0;
            for (int i = 0; i < att_table.length; i++) {
                colSum += att_table[i][j];
                if (first) {
                    rowSums[i] = Arrays.stream(att_table[i]).sum();
                }
            }
            colSums[j] = colSum;
            first = false;
        }

        int colSumsSum = Arrays.stream(colSums).sum();
        double chi = 0;

        for (int i = 0; i < att_table.length; i++) {
            for (int j = 0; j < att_table[i].length; j++) {
                exp_values[i][j] = rowSums[i] * (colSums[j] / (float)colSumsSum);
                chi += Math.pow(att_table[i][j] - exp_values[i][j], 2) / exp_values[i][j];
            }
        }

        return chi;
    }

    /**
     * Measures the chi-squared statistic with Yates correction for
     * the contingency table according to the formula:
     * x^2=sum[i=1->r](sum[j=1->c]((oij-eij-0.5)^2/eij)))
     *
     * @param att_table the contingency table for the attribute
     * @return the chi-squared statistic with yates corr. as a double
     */
    public static double measureChiSquaredYates(int[][] att_table) {

        int[] colSums = new int[att_table[0].length];
        int[] rowSums = new int[att_table.length];
        double[][] exp_values = new double[att_table.length][att_table[0].length];
        boolean first = true;

        for (int j = 0; j < att_table[0].length; j++) {
            int colSum = 0;
            for (int i = 0; i < att_table.length; i++) {
                colSum += att_table[i][j];
                if (first) {
                    rowSums[i] = Arrays.stream(att_table[i]).sum();
                }
            }
            colSums[j] = colSum;
            first = false;
        }

        int colSumsSum = Arrays.stream(colSums).sum();
        double chiYates = 0;

        for (int i = 0; i < att_table.length; i++) {
            for (int j = 0; j < att_table[i].length; j++) {
                exp_values[i][j] = rowSums[i] * (colSums[j] / (float)colSumsSum);
                chiYates += Math.pow(att_table[i][j] - exp_values[i][j] - 0.5, 2) / exp_values[i][j];
            }
        }

        return chiYates;
    }

    public static double logBase2(double x) {
        return (Math.log(x) / Math.log(2));
    }

    public static void main (String[] args) {
        int[][] testTable = {{3, 2}, {3, 4}};
        //int[] classCounts = {3, 4};

        //double ent = calcEntropy(classCounts);

        double infGain = measureInformationGain(testTable);
        double gini = measureGini(testTable);
        double chi = measureChiSquared(testTable);
        double chiYates = measureChiSquaredYates(testTable);

        //System.out.println("entropy = " + ent);
        System.out.println("measure information gain for headache " +
                "splitting diagnosis = " + infGain);
        System.out.println("measure gini index for headache splitting " +
                "diagnosis = " + gini);
        System.out.println("measure chi-squared statistic for headache " +
                "splitting diagnosis = " + chi);
        System.out.println("measure chi-squared with Yates correction for " +
                "headache splitting diagnosis = " + chiYates);
    }
}
