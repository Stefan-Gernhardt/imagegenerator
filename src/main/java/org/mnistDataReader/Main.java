package org.mnistDataReader;

import org.mnistDataReader.MnistMatrix;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        MnistMatrix[] mnistTrainMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        printMnistMatrix(mnistTrainMatrix[0]);
        MnistMatrix[] mnistValidateMatrix = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        printMnistMatrix(mnistValidateMatrix[0]);
    }

    private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}
