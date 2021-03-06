/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Id3.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 */

package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;
import java.util.Vector;

/**
 * CMP-6002B Machine Learning Classification with Decision Trees
 *
 * Adaptation of the Id3 Weka Classifier for use in Machine Learning Coursework (6002B)
 *
 * Ref: Id3.java, Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 <!-- globalinfo-start -->
 * Class for constructing an unpruned decision tree based on the ID3 algorithm.
 * Can only deal with nominal attributes. No missing values allowed.
 * Empty leaves may result in unclassified instances. For more information see: <br/>
 * <br/>
 * R. Quinlan (1986). Induction of decision trees. Machine Learning. 1(1):81-106.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Quinlan1986,
 *    author = {R. Quinlan},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {81-106},
 *    title = {Induction of decision trees},
 *    volume = {1},
 *    year = {1986}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 6404 $
 *
 * @author Alex Middlemiss, 100219171, exb17gxu
 * @version 1.0, 21/03/2021
 */
public class ID3Coursework extends AbstractClassifier
        implements TechnicalInformationHandler, Sourcable {

    /** for serialization */
    static final long serialVersionUID = -2693678647096322561L;

    /** The node's successors. */
    private ID3Coursework[] m_Successors;

    /** Attribute used for splitting. */
    private Attribute m_Attribute;

    /** Class value if node is leaf. */
    private double m_ClassValue;

    /** Class distribution if node is leaf. */
    private double[] m_Distribution;

    /** Class attribute of dataset. */
    private Attribute m_ClassAttribute;
    private AttributeSplitMeasure attSplit = new IGAttributeSplitMeasure();

    /**
     * Returns a string describing the classifier.
     *
     * @return a description suitable for the GUI.
     */
    public String globalInfo() {
        return  "Class for constructing an unpruned decision tree based on the ID3 "
                + "algorithm. Can only deal with nominal attributes. No missing values "
                + "allowed. Empty leaves may result in unclassified instances. For more "
                + "information see: \n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "R. Quinlan");
        result.setValue(Field.YEAR, "1986");
        result.setValue(Field.TITLE, "Induction of decision trees");
        result.setValue(Field.JOURNAL, "Machine Learning");
        result.setValue(Field.VOLUME, "1");
        result.setValue(Field.NUMBER, "1");
        result.setValue(Field.PAGES, "81-106");

        return result;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Builds Id3 decision tree classifier.
     *
     * @param data the training data
     * @exception Exception if classifier can't be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        data = new Instances(data);
        data.deleteWithMissingClass();

        makeTree(data);
    }

    /**
     * Method for building an Id3 tree.
     *
     * @param data the training data
     * @exception Exception if decision tree can't be built successfully
     */
    private void makeTree(Instances data) throws Exception {

        // Check if no instances have reached this node.
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = Utils.missingValue();
            m_Distribution = new double[data.numClasses()];
            return;
        }

        // Compute attribute with maximum information gain.
        double[] infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = attSplit.computeAttributeQuality(data, att);
        }
        m_Attribute = data.attribute(Utils.maxIndex(infoGains));

        // Make leaf if information gain is zero.
        // Otherwise create successors.
        if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
            m_Attribute = null;
            m_Distribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                m_Distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
            m_ClassAttribute = data.classAttribute();
        } else {
            Instances[] splitData = attSplit.splitData(data, m_Attribute);
            m_Successors = new ID3Coursework[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                m_Successors[j] = new ID3Coursework();
                m_Successors[j].makeTree(splitData[j]);
            }
        }
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    public double classifyInstance(Instance instance)
            throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, please.");
        }
        if (m_Attribute == null) {
            return m_ClassValue;
        } else {
            return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
        }
    }

    /**
     * Computes class distribution for instance using decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    public double[] distributionForInstance(Instance instance)
            throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Id3: no missing values, please.");
        }
        if (m_Attribute == null) {
            return m_Distribution;
        } else {
            return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
        }
    }

    /**
     * Prints the decision tree using the private toString method from below.
     *
     * @return a textual description of the classifier
     */
    public String toString() {
        if ((m_Distribution == null) && (m_Successors == null)) {
            return "Id3: No model built yet.";
        }
        return "Id3\n\n" + toString(0);
    }

    /**
     * Outputs a tree at a certain level.
     *
     * @param level the level at which the tree is to be printed
     * @return the tree as string at the given level
     */
    private String toString(int level) {
        StringBuilder text = new StringBuilder();

        if (m_Attribute == null) {
            if (Utils.isMissingValue(m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": ").append(m_ClassAttribute.value((int) m_ClassValue));
            }
        } else {
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(m_Attribute.name()).append(" = ").append(m_Attribute.value(j));
                text.append(m_Successors[j].toString(level + 1));
            }
        }
        return text.toString();
    }

    /**
     * Adds this tree recursively to the buffer.
     *
     * @param id the unqiue id for the method
     * @param buffer the buffer to add the source code to
     * @return the last ID being used
     * @throws Exception if something goes wrong
     */
    protected int toSource(int id, StringBuffer buffer) throws Exception {
        int result, i, newID;
        StringBuffer[] subBuffers;

        buffer.append("\n");
        buffer.append("  protected static double node").append(id).append("(Object[] i) {\n");

        // leaf?
        if (m_Attribute == null) {
            result = id;
            if (Double.isNaN(m_ClassValue)) {
                buffer.append("    return Double.NaN;");
            } else {
                buffer.append("    return ").append(m_ClassValue).append(";");
            }
            if (m_ClassAttribute != null) {
                buffer.append(" // ").append(m_ClassAttribute.value((int) m_ClassValue));
            }
            buffer.append("\n");
            buffer.append("  }\n");
        } else {
            buffer.append("    checkMissing(i, ").append(m_Attribute.index()).append(");\n\n");
            buffer.append("    // ").append(m_Attribute.name()).append("\n");

            // subtree calls
            subBuffers = new StringBuffer[m_Attribute.numValues()];
            newID = id;
            for (i = 0; i < m_Attribute.numValues(); i++) {
                newID++;
                buffer.append("    ");
                if (i > 0) {
                    buffer.append("else ");
                }
                buffer.append("if (((String) i[").append(m_Attribute.index())
                        .append("]).equals(\"").append(m_Attribute.value(i)).append("\"))\n");
                buffer.append("      return node").append(newID).append("(i);\n");

                subBuffers[i] = new StringBuffer();
                newID = m_Successors[i].toSource(newID, subBuffers[i]);
            }
            buffer.append("    else\n");
            buffer.append("      throw new IllegalArgumentException(\"Value '\" + i[")
                    .append(m_Attribute.index()).append("] + \"' is not allowed!\");\n");
            buffer.append("  }\n");

            // output subtree code
            for (i = 0; i < m_Attribute.numValues(); i++) {
                buffer.append(subBuffers[i].toString());
            }
            subBuffers = null;
            result = newID;
        }
        return result;
    }

    /**
     * Returns a string that describes the classifier as source.
     * The classifier will be contained in a class with the given name
     * (there may be auxiliary classes), and will contain a method with the signature:
     * <pre><code>
     * public static double classify(Object[] i);
     * </code></pre>
     * where the array <code>i</code> contains elements that are either
     * Double, String, with missing values represented as null.
     * The generated code is public domain and comes with no warranty. <br/>
     * Note: works only if class attribute is the last attribute in the dataset.
     *
     * @param className the name that should be given to the source class
     * @return the object source described by a string
     * @throws Exception if the source can't be computed
     */
    public String toSource(String className) throws Exception {
        StringBuffer result;
        int id;
        result = new StringBuffer();
        result.append("class ").append(className).append(" {\n");
        result.append("  private static void checkMissing(Object[] i, int index) {\n");
        result.append("    if (i[index] == null)\n");
        result.append("      throw new IllegalArgumentException(\"Null values "
                + "are not allowed!\");\n");
        result.append("  }\n\n");
        result.append("  public static double classify(Object[] i) {\n");
        id = 0;
        result.append("    return node").append(id).append("(i);\n");
        result.append("  }\n");
        toSource(id, result);
        result.append("}\n");
        return result.toString();
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 6404 $");
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(4);

        newVector.addElement(new Option(
                "\tIf set, classifier is run using Information Gain\n"
                        + "\tas the attribute selection measure.",
                "I", 0, "-I"));
        newVector.addElement(new Option(
                "\tIf set, classifier is run using Gini Index\n"
                        + "\tas the attribute selection measure.",
                "G", 0, "-G"));
        newVector.addElement(new Option(
                "\tIf set, classifier is run using Chi Squared\n"
                        + "\tas the attribute selection measure.",
                "C", 0, "-C"));
        newVector.addElement(new Option(
                "\tIf set, classifier is run using Chi Squared\n"
                        + "\t with Yates as the attribute selection measure.",
                "Y", 0, "-Y"));

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        setAtt(Utils.getFlag('I', options), 'I');
        setAtt(Utils.getFlag('G', options), 'G');
        setAtt(Utils.getFlag('C', options), 'C');
        setAtt(Utils.getFlag('Y', options), 'Y');
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        String[] options = new String[1];
        options[0] = getAtt().substring(0, 1);
        return options;
    }

    /**
     * Set attribute selection measure.
     *
     * @param attPresent true if attribute flag present
     * @param att char representing the attribute
     */
    public void setAtt(boolean attPresent, char att) {
        if (attPresent) {
            switch (att) {
                case 'I': attSplit = new IGAttributeSplitMeasure(); break;
                case 'G': attSplit = new GiniAttributeSplitMeasure(); break;
                case 'C': attSplit = new ChiSquaredAttributeSplitMeasure(); break;
                case 'Y': attSplit = new ChiSquaredAttributeSplitMeasure(true); break;
                default: attSplit = new IGAttributeSplitMeasure();
            }
        }
    }

    /**
     * Get which attribute measure selected.
     *
     * @return String attribute measure
     */
    public String getAtt() {
        return attSplit.toString();
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier
     */
    public static void main(String[] args) {
        runClassifier(new ID3Coursework(), args);
    }
}
