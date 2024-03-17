package org.deeplearning;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class Util {

    public static INDArray setIndArrayToConstant(INDArray indArray, double d) {

        for (int r = 0; r < indArray.rows(); r++) {
            for (int c = 0; c < indArray.columns(); c++) {
                indArray.put(r, c, d);
            }
        }

        return indArray;
    }

    public static INDArray setIndArrayWithGaussDistribution(INDArray indArray, double mean, double std, double lowerBorder, double upperBorder) {
        Random random = new Random();

        for (int r = 0; r < indArray.rows(); r++) {
            for (int c = 0; c < indArray.columns(); c++) {
                double value = random.nextGaussian() * mean + std;
                if (value > upperBorder) value = upperBorder;
                if (value < lowerBorder) value = lowerBorder;
                indArray.put(r, c, value);
            }
        }

        return indArray;
    }

    public static void setIndArrayWithGaussDistribution(INDArray indArray, INDArray mean, INDArray std, double lowerBorder, double upperBorder) {
        Random random = new Random();

        for (int r = 0; r < indArray.rows(); r++) {
            for (int c = 0; c < indArray.columns(); c++) {
                double value = random.nextGaussian() * mean.getDouble(0, c) + std.getDouble(0, c);
                if (value > upperBorder) value = upperBorder;
                if (value < lowerBorder) value = lowerBorder;
                indArray.put(r, c, value);
            }
        }
    }

    public static void computeMeanAndStd(INDArray eliteGeneticINDArray, INDArray geneticMeanINDArray, INDArray geneticSTDINDArray) {
        for (int c = 0; c < eliteGeneticINDArray.columns(); c++) {

            double mean = 0;
            for (int r = 0; r < eliteGeneticINDArray.rows(); r++) {
                // System.out.println("value eliteIndArray row: " + r + " col: " + c + " value: " + eliteGeneticINDArray.getDouble(r, c));
                mean = mean + eliteGeneticINDArray.getDouble(r, c);
            }
            mean = mean / eliteGeneticINDArray.rows();
            // System.out.println("mean: " + mean);
            geneticMeanINDArray.put(0, c, mean);

            double sumOfSquares = 0.0;
            for (int r = 0; r < eliteGeneticINDArray.rows(); r++) {
                sumOfSquares += Math.pow(eliteGeneticINDArray.getDouble(r, c) - mean, 2);
            }
            double variance = sumOfSquares / eliteGeneticINDArray.rows();
            double stdDev = Math.sqrt(variance);
            geneticSTDINDArray.put(0, c, stdDev);
        }
    }

    public static void printIndarray(INDArray indArray) {
        System.out.println("----------------------------");
        System.out.println("INDArray");
        for (int row = 0; row < indArray.rows(); row++) {
            for (int col = 0; col < indArray.columns(); col++) {
                double value = indArray.getDouble(row, col);
                System.out.print("" + value + ", ");
            }
            System.out.println();
        }
    }
}
