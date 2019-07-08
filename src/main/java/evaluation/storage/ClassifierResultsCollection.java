/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package evaluation.storage;

import experiments.DataSets;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.function.Function;
import utilities.DebugPrinting;
import utilities.ErrorReport;

/**
 * Essentially a loader for many results over a given set of classifiers, datasets, folds, and splits
 * 
 * Usage: 
 *      Construct the object
 *      Set classifiers, datasets, folds and splits to read at MINIMUM
 *      Set any optional settings on how to read and store the results
 *      Call load()
 *          Either use the big old ClassifierResults[][][][] returned, or interact with the 
 *          collection via the slice or getInfo methods 
 * 
 *      Afterwards, optionally use the slice...() methods to get subsets of the results 
 *          already loaded into memory
 *      Or use the getInfo(...) method to get a particular stat from each results object
 *          getAccuracies() wraps the accuracies getter as a shortcut/example 
 * 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ClassifierResultsCollection implements DebugPrinting {
    
    /**
     * ClassifierResults[split][classifier][dataset][fold]
     * Split taken to be first dimension for easy retrieval of a single split if only one is loaded
     * and you want results in the 3d form [classifier][dataset][fold]
     */
    public ClassifierResults[][][][] allResults;

    public int numDatasets;
    public String[] datasetNamesInStorage;
    public String[] datasetNamesInOutput;
    
    public int numClassifiers;
    public String[] classifierNamesInStorage;
    public String[] classifierNamesInOutput;
    
    public int numFolds;
    public int[] folds;
    
    public int numSplits;
    public String[] splits;
    
    
    /**
     * A path to a directory containing all the classifierNamesInStorage directories 
     * with the results, in format {baseReadPath}/{classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv
     */
    public String baseReadPath;
    
    
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found for it 
     * 
     * defaults to true
     */
    private boolean cleanResults = true;

    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     * 
     * defaults to true
     */
    private boolean testResultsOnly = true;
    
    /**
     * if true, the returned lists are guaranteed to be of size numClassifiers*numDsets*numFolds*2,
     * but entries may be null;
     * 
     * defaults to false
     */
    public boolean allowMissingResults = false;

    /**
     * if true, will fill in missing probability distributions with one-hot vectors
     * for files read in that are missing them. intended for very old files, where you still 
     * want to calc auroc etc (metrics that need dists) for all the other classifiers 
     * that DO provide them, but also want to compare e.g accuracy with classifier that don't
     * 
     * defaults to false
     */
    private boolean ignoreMissingDistributions = false;
    
    public ClassifierResultsCollection() {
        
    }
    
    /**
     * Creates complete copy of the other collection, but keeps the subResults instead of the 
     * other's results. Intended for use when slicing, and then manually edit the particular 
     * bit of meta info that was sliced 
     */
    private ClassifierResultsCollection(ClassifierResultsCollection other, ClassifierResults[][][][] subResults) {
        this.allResults = subResults;
        
        this.numDatasets = other.numDatasets;
        this.datasetNamesInStorage = other.datasetNamesInStorage;
        this.datasetNamesInOutput = other.datasetNamesInOutput;

        this.numClassifiers = other.numClassifiers;
        this.classifierNamesInStorage = other.classifierNamesInStorage;
        this.classifierNamesInOutput = other.classifierNamesInOutput;

        this.numFolds = other.numFolds;
        this.folds = other.folds;

        this.numSplits = other.numSplits;
        this.splits = other.splits;

        this.baseReadPath = other.baseReadPath;

        this.cleanResults = other.cleanResults;
        this.testResultsOnly = other.testResultsOnly;
        this.allowMissingResults = other.allowMissingResults;
        this.ignoreMissingDistributions = other.ignoreMissingDistributions;
    }
    
    /**
     * Sets the number folds/resamples to read in for each classifier/dataset. 
     * Will create a range of fold ids from 0(inclusive) to maxFolds(exclusive)
     */
    public void setFolds(int maxFolds) { 
        setFolds(0, maxFolds);
    }
    
    /**
     * Sets the folds/resamples to read in for each classifier/dataset
     * Will create a range of fold ids from minFolds(inclusive) to maxFolds(exclusive)
     */
    public void setFolds(int minFolds, int maxFolds) { 
        folds = buildRange(minFolds, maxFolds);
    }
    
    /**
     * Sets the specific folds/resamples to read in for each classifier/dataset, 
     * to be used if the folds wanted to not lie in a continuous range for example
     */
    public void setFolds(int[] foldIds) {
        this.folds = foldIds;
        this.numFolds = foldIds.length;                
    }
    
    
    /**
     * Sets the classifiers to be read in. Names must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void setClassifiers(String[] classifierNames) {
        setClassifiers(classifierNames, classifierNames);
    }
    
    /**
     * Sets the classifiers to be read in. classifierNamesInStorage must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist,
     * while classifierNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void setClassifiers(String[] classifierNamesInStorage, String[] classifierNamesInOutput) {
        if (classifierNamesInStorage.length != classifierNamesInOutput.length)
            throw new IllegalArgumentException("Classifier names lengths not equal, "
                    + "classifierNamesInStorage.length="+classifierNamesInStorage.length
                    + " classifierNamesInOutput.length="+classifierNamesInOutput.length);
        
        this.numClassifiers = classifierNamesInOutput.length;
        this.classifierNamesInStorage = classifierNamesInStorage;
        this.classifierNamesInOutput = classifierNamesInOutput;
        
    }
    
    /**
     * Sets the datasets to be read in for each classifier. Names must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each classifier,
     */
    public void setDatasets(String[] datasetNames) {
        setDatasets(datasetNames, datasetNames);
    }
    
    /**
     * Sets the datasets to be read in for each classifier. datasetNamesInStorage must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each classifier,
     * while classifierNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void setDatasets(String[] datasetNamesInStorage, String[] datasetNamesInOutput) {
        if (datasetNamesInStorage.length != datasetNamesInStorage.length)
            throw new IllegalArgumentException("Classifier datasetNamesInOutput lengths not equal, "
                    + "datasetNamesInStorage.length="+datasetNamesInStorage.length
                    + " datasetNamesInOutput.length="+datasetNamesInOutput.length);
        
        this.numDatasets = datasetNamesInStorage.length;
        this.datasetNamesInStorage = datasetNamesInStorage;
        this.datasetNamesInOutput = datasetNamesInOutput;
    }
    
    /**
     * Set to look for train fold files only for each classifier/dataset/fold
     */
    public void setSplit_Train() {
        setSplit("train");
    }

    /**
     * Sets to look for test fold files only for each classifier/dataset/fold
     */
    public void setSplit_Test() {
        setSplit("test");
    }
    
    /**
     * Sets to look for train AND test fold files for each classifier/dataset/fold
     */
    public void setSplit_TrainTest() {
        setSplits(new String[] { "train", "test" });
    }
    
    /**
     * Sets to look for a particular dataset split, test and test are currently 
     * the only options generated by e.g. Experiments.java. In the future, things 
     * like validation, cvFoldX, etc might be possible
     */
    public void setSplit(String split) {
        this.splits = new String[] { split };
        this.numSplits = 1;
    }
    
    /**
     * Sets to look for a particular dataset split, test and test are currently 
     * the only options generated by e.g. Experiments.java. In the future, things 
     * like validation, cvFoldX, etc might be possible
     */
    public void setSplits(String[] splits) {
        this.splits = splits;
        this.numSplits = splits.length;
    }
    
    /**
     * Sets the path to a directory containing all the classifierNamesInStorage directories 
     * with the results, in format {baseReadPath}/{classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv
     */
    public void setBaseReadPath(String baseReadPath) {
        baseReadPath.replace("\\", "/");
        if (baseReadPath.charAt(baseReadPath.length()-1) != '/')
            baseReadPath += "/";
        
        this.baseReadPath = baseReadPath;
    }
    
    /**
     * Sets the path to a directory containing all the classifierNamesInStorage directories 
     * with the results, in format {baseReadPath}/{classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv
     */
    public void setBaseReadPath(File baseReadPath) {
        setBaseReadPath(baseReadPath.getAbsoluteFile());
    }
    
    private void confirmMinimalInfoGivenAndValid() throws Exception {
        ErrorReport err = new ErrorReport("Required results collection info missing:\n");
        
        if (baseReadPath == null)
            err.log("\tBase path to read results from not set\n");
        else if (!(new File(baseReadPath).exists()))
            err.log("\tBase path to read results from cannot be found: " + baseReadPath + "\n");
                    
        if (classifierNamesInStorage == null || classifierNamesInStorage.length == 0)
            err.log("\tClassifiers to read not set\n");
        
        if (datasetNamesInStorage == null || datasetNamesInStorage.length == 0)
            err.log("\tDatasets to read not set\n");
        
        if (folds == null || folds.length == 0)
            err.log("\tFolds to read not set\n");
        
        if (splits == null || splits.length == 0)
            err.log("\tSplits to read not set\n");
        
        err.throwIfErrors();
    }
    
    private static int[] buildRange(int minFolds, int maxFolds) { 
        int[] folds = new int[maxFolds - minFolds];
        
        int c = minFolds;
        for (int i = 0; i < maxFolds - minFolds; i++, c++)
            folds[i] = c;
        
        return folds;
    }
    
    private static int find(String[] arr, String k) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i].equals(k))
                return i;
        return -1;
    }
    
    private static int find(int[] arr, int k) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i] == (k))
                return i;
        return -1;
    }
    
    private static String buildFileName(String baseReadPath, String classifier, String dataset, String split, int fold) { 
        return baseReadPath + classifier + "/Predictions/" + dataset + "/" + split + "Fold" + fold + ".csv";
    }
    
    private static void throwSliceError(String str, String[] arr, String key) throws Exception {
        throw new Exception("SLICE ERROR: Attempted to slice " + str + " by " + key + " but that does not exist in " + Arrays.toString(arr));
    }
    
    private static void throwSliceError(String str, int[] arr, int key) throws Exception {
        throw new Exception("SLICE ERROR: Attempted to slice " + str + " by " + key + " but that does not exist in " + Arrays.toString(arr));
    }
    
    public ClassifierResults[][][][] load() throws Exception { 
        confirmMinimalInfoGivenAndValid();
        
        ErrorReport masterError = new ErrorReport("Results files not found:\n");

        allResults = new ClassifierResults[numSplits][numClassifiers][numDatasets][numFolds];
     
        //train files may be produced via TrainAccuracyEstimate, older code
        //while test files likely by experiments, but still might be a very old file
        //so having separate checks for each.
        boolean ignoringDistsFirstTime = true;
        
        for (int c = 0; c < numClassifiers; c++) {
            String classifierStorage = classifierNamesInStorage[c];
            String classifierOutput = classifierNamesInOutput[c];
            printlnDebug(classifierStorage + "(" + classifierOutput + ") reading");
            
            try {

                int totalFnfs = 0;
                ErrorReport perClassifierError = new ErrorReport("FileNotFoundExceptions thrown:\n");

                for (int d = 0; d < numDatasets; d++) {
                    String datasetStorage = datasetNamesInStorage[d];
                    String datasetOutput = datasetNamesInOutput[d];
                    printlnDebug("\t" + datasetStorage + "(" + datasetOutput + ") reading");

                    for (int f = 0; f < numFolds; f++) {
                        int fold = folds[f];
                        printlnDebug("\t\t" + fold + " reading");

                        for (int s = 0; s < numSplits; s++) {
                            String split = splits[s];     
                            printlnDebug("\t\t\t" + split + " reading");

                            String fileName = buildFileName(baseReadPath, classifierStorage, datasetStorage, split, fold); 
                            try {
                                allResults[s][c][d][f] = new ClassifierResults(fileName);
                                if (ignoreMissingDistributions) {
                                    boolean wasMissing = allResults[s][c][d][f].populateMissingDists();
                                    if (wasMissing && ignoringDistsFirstTime) {
                                        System.out.println("---------Probability distributions missing, but ignored: " 
                                                + classifierStorage + " - " + datasetStorage + " - " + f + " - train");
                                        ignoringDistsFirstTime = false;
                                    }
                                }
                                allResults[s][c][d][f].findAllStatsOnce();
                                if (cleanResults)
                                    allResults[s][c][d][f].cleanPredictionInfo();
                            } catch (FileNotFoundException ex) {
                                perClassifierError.log(fileName + "\n");
                                totalFnfs++;
                            }

                            printlnDebug("\t\t\t" + split + " successfully read in");
                        }
                        printlnDebug("\t\t" + fold + " successfully read in");
                    }
                    printlnDebug("\t" + datasetStorage + "(" + datasetOutput + ") successfully read in");
                }

                if (!perClassifierError.isEmpty())
                    perClassifierError.log("Total num errors for " + classifierStorage + ": " + totalFnfs);
                perClassifierError.throwIfErrors();
                printlnDebug(classifierStorage + "(" + classifierOutput + ") successfully read in");
            } catch (Exception e) {
                masterError.log("Classifier Errors: " + classifierNamesInStorage[c] + "\n" + e);
            }
        }
        
        masterError.throwIfErrors();
        
        return allResults;
    }
    
    
    
    
    public ClassifierResultsCollection sliceSplit(String split) throws Exception { 
        return sliceSplits(new String[] { split });
    }
        
    public ClassifierResultsCollection sliceSplits(String[] splitsToSlice) throws Exception {
        //perform existence checks before allocating the mem
        for (String split : splitsToSlice)
            if (find(splits, split) == -1)
                throwSliceError("splits", splits, split);
        
        //copy across the results, for splits it's nice and easy
        ClassifierResults[][][][] subResults = new ClassifierResults[splitsToSlice.length][][][];
        for (int sts = 0; sts < splitsToSlice.length; sts++) {
            int sidOrig = find(splits, splitsToSlice[sts]); //know it exists, did checks above
            subResults[sts] = this.allResults[sidOrig];
        }
        
        //copy across the meta info to neew collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setSplits(splitsToSlice); //checking the particular meta info sliced
        return newCol;
    }
        
    public ClassifierResultsCollection sliceClassifier(String classifier) throws Exception { 
        return sliceClassifiers(new String[] { classifier });
    }
    
    public ClassifierResultsCollection sliceClassifiers(String[] classifiersToSlice) throws Exception { 
        //perform existence checks before allocating the mem
        for (String classifier : classifiersToSlice)
            if (find(classifierNamesInStorage, classifier) == -1)
                throwSliceError("classifiers", classifierNamesInStorage, classifier);
                
        //copy across the results
        ClassifierResults[][][][] subResults = new ClassifierResults[numSplits][classifiersToSlice.length][][];
        for (int s = 0; s < numSplits; s++) {
            for (int cts = 0; cts < classifiersToSlice.length; cts++) {
                int cidOrig = find(classifierNamesInStorage, classifiersToSlice[cts]); //know it exists, did checks above
                subResults[s][cts] = this.allResults[s][cidOrig];
            }
        }
        
        //copy across the meta info to neew collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setClassifiers(classifiersToSlice); //checking the particular meta info sliced
        return newCol;
    }
    
    public ClassifierResultsCollection sliceDataset(String dataset) throws Exception { 
        return sliceDatasets(new String[] { dataset });
    }
    
    public ClassifierResultsCollection sliceDatasets(String[] datasetsToSlice) throws Exception { 
        //perform existence checks before allocating the mem
        for (String dataset : datasetsToSlice)
            if (find(datasetNamesInStorage, dataset) == -1)
                throwSliceError("datasets", datasetNamesInStorage, dataset);
                
        //copy across the results
        ClassifierResults[][][][] subResults = new ClassifierResults[numSplits][numClassifiers][datasetsToSlice.length][];
        for (int s = 0; s < numSplits; s++) {
            for (int c = 0; c < numClassifiers; c++) {
                for (int dts = 0; dts < datasetsToSlice.length; dts++) {
                    int didOrig = find(datasetNamesInStorage, datasetsToSlice[dts]); //know it exists, did checks above
                    subResults[s][c][dts] = this.allResults[s][c][didOrig];
                }
            }
        }
        
        //copy across the meta info to neew collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setDatasets(datasetsToSlice); //checking the particular meta info sliced
        return newCol;
    }
    
    public ClassifierResultsCollection sliceFold(int fold) throws Exception { 
        return sliceFolds(new int[] { fold });
    }
    
    public ClassifierResultsCollection sliceFolds(int minFolds, int maxFolds) throws Exception { 
        return sliceFolds(buildRange(minFolds, maxFolds));
    }
    
    public ClassifierResultsCollection sliceFolds(int[] foldsToSlice) throws Exception { 
        //perform existence checks before allocating the mem
        for (int fold : foldsToSlice)
            if (find(folds, fold) == -1)
                throwSliceError("folds", folds, fold);
                
        //copy across the results
        ClassifierResults[][][][] subResults = new ClassifierResults[numSplits][numClassifiers][numDatasets][foldsToSlice.length];
        for (int s = 0; s < numSplits; s++) {
            for (int c = 0; c < numClassifiers; c++) {
                for (int d = 0; d < numDatasets; d++) {
                    for (int fts = 0; fts < foldsToSlice.length; fts++) {
                        int fidOrig = find(folds, foldsToSlice[fts]); //know it exists, did checks above
                        subResults[s][c][d][fts] = this.allResults[s][c][d][fidOrig];
                    }
                }
            }
        }
        
        //copy across the meta info to neew collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setFolds(foldsToSlice); //checking the particular meta info sliced
        return newCol;
    }

    
    

    
    
    
    
    
    public double[][][][] getAccuracies() {
        return getInfo_double(ClassifierResults.GETTER_Accuracy);
    }

    //todo make generic
    public double[][][][] getInfo_double(Function<ClassifierResults, Double> getter) {
        double[][][][] info = new double[numSplits][numClassifiers][numDatasets][numFolds];
        for (int i = 0; i < numSplits; i++)
            for (int j = 0; j < numClassifiers; j++)
                for (int k = 0; k < numDatasets; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allResults[i][j][k][l]);
        return info;
    }
    
    //todo make generic
    public String[][][][] getInfo_String(Function<ClassifierResults, String> getter) {
        String[][][][] info = new String[numSplits][numClassifiers][numDatasets][numFolds];
        for (int i = 0; i < numSplits; i++)
            for (int j = 0; j < numClassifiers; j++)
                for (int k = 0; k < numDatasets; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allResults[i][j][k][l]);
        return info;
    }
    
    
    public static void main(String[] args) throws Exception {
        ClassifierResultsCollection col = new ClassifierResultsCollection();
        col.setBaseReadPath("C:/JamesLPHD/CAWPEExtension/Results/");
        col.setClassifiers(new String[] { "Logistic", "SVML", "MLP" });
        col.setDatasets(Arrays.copyOfRange(DataSets.ReducedUCI, 0, 5));
        col.setFolds(10);
        col.setSplit_Test();
        
        ClassifierResults[][][][] res = col.load();
        System.out.println(res.length);
        System.out.println(res[0].length);
        System.out.println(res[0][0].length);
        System.out.println(res[0][0][0].length);        
        System.out.println(res[0][0][0][0].getAcc());      
        
        double[][][][] accs = col.getAccuracies();
        System.out.println(accs.length);
        System.out.println(accs[0].length);
        System.out.println(accs[0][0].length);
        System.out.println(accs[0][0][0].length);        
        System.out.println(accs[0][0][0][0]);  
    }
}
