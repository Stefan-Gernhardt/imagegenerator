package org.deeplearning;

import org.mnistDataReader.MnistDataReader;
import org.mnistDataReader.MnistMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class MnistData {
    public static final int IMG_SIZE = 28;
    private MnistMatrix[] mnistTrainMatrix;

    private MnistMatrix[] mnistValidationMatrix;

    private INDArray trainingInputs = null;

    private INDArray trainingOutputs = null;

    private INDArray validationInputs = null;

    private INDArray validationOutputs = null;


    public MnistData() {
        loadData();
    }

    public void loadData() {
        try {
            mnistTrainMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
            mnistValidationMatrix = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        System.out.println("number of train images: " + mnistTrainMatrix.length);

        trainingInputs = Nd4j.zeros(mnistTrainMatrix.length, IMG_SIZE * IMG_SIZE);
        trainingOutputs = Nd4j.zeros(mnistTrainMatrix.length, 10);

        for (int i = 0; i < mnistTrainMatrix.length; i++) {
            for (int r = 0; r < IMG_SIZE; r++) {
                for (int c = 0; c < IMG_SIZE; c++) {
                    double d = mnistTrainMatrix[i].getValue(r, c) / 255.0;
                    trainingInputs.put(i, r * IMG_SIZE + c, d);
                }
            }
        }

        for (int i = 0; i < mnistTrainMatrix.length; i++) {
            trainingOutputs.put(i, mnistTrainMatrix[i].getLabel(), 1.0);
        }

        validationInputs = Nd4j.zeros(mnistValidationMatrix.length, IMG_SIZE * IMG_SIZE);
        validationOutputs = Nd4j.zeros(mnistValidationMatrix.length, 10);

        for (int i = 0; i < mnistValidationMatrix.length; i++) {
            for (int r = 0; r < IMG_SIZE; r++) {
                for (int c = 0; c < IMG_SIZE; c++) {
                    double d = mnistValidationMatrix[i].getValue(r, c) / 255.0;
                    validationInputs.put(i, r * IMG_SIZE + c, d);
                }
            }
        }

        for (int i = 0; i < mnistValidationMatrix.length; i++) {
            validationOutputs.put(i, mnistValidationMatrix[i].getLabel(), 1.0);
        }
    }

    public MnistMatrix[] getMnistTrainMatrix() {
        return mnistTrainMatrix;
    }

    public MnistMatrix[] getMnistValidationMatrix() {
        return mnistValidationMatrix;
    }

    public INDArray getTrainingInputs() {
        return trainingInputs;
    }

    public INDArray getTrainingOutputs() {
        return trainingOutputs;
    }

    public INDArray getValidationInputs() {
        return validationInputs;
    }

    public INDArray getValidationOutputs() {
        return validationOutputs;
    }

    public validateResult validate(Discriminator discriminator, int digit) {
        int numberPrintOutputs = 0;
        int countData = 0;

        int winner = 0;
        int loose = 0;
        double winRate = 0;

        for(int i=0; i<mnistValidationMatrix.length; i++) {
            int label = getLabelValidationData(i);
            if(label == digit) {
                countData++;

                if(i < numberPrintOutputs) System.out.println("" + validationOutputs.getRow(i));
                if(i < numberPrintOutputs) System.out.println("label: " + label);

                INDArray evalInputRow = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
                evalInputRow.putRow(0, validationInputs.getRow(i));

                boolean printout = false;
                if(i < numberPrintOutputs) printout = true;

                EvalResult evalResult = discriminator.askModel(evalInputRow, printout);
                if(i < numberPrintOutputs) System.out.println(evalResult.label);

                if(evalResult.label == Discriminator.IS_TRUTH) {
                    winner++;
                    winRate = winRate + evalResult.value;
                    // System.out.println("count data: " + countData + "  winrate average: " + winRate / countData);
                }
                else {
                    loose++;
                }
            }
        }

        System.out.println("countData: " + countData + " / " + mnistValidationMatrix.length);
        System.out.println("win: "   + 1.0*winner  /  countData);

        return new validateResult(1.0*winner / countData, winRate / winner);
    }

    private int getLabelValidationData(int i) {
        // System.out.println("" + validationOutputs.getRow(i));
        return mnistValidationMatrix[i].getLabel();
    }
}
