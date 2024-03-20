package org.deeplearning;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.mnistDataReader.MnistDataReader;
import org.mnistDataReader.MnistMatrix;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.List;

import static org.deeplearning.MnistData.IMG_SIZE;

// Sources:
// https://gist.github.com/tomthetrainer/7cb2fbc14a5c631a567a98c3134f7dd6
// https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/android/image-classification
// https://medium.com/mlearning-ai/neural-networks-getting-started-with-eclipse-deeplearning4j-897f3662832b

public class MnistSimple {
    public static final String MNIST_SIMPLE_FILE_NAME = "MnistSimple";

    public static final String MNIST_SIMPLE_FILE_NAME_INIT_VALUES_ARE_ZERO = "MnistSimple196";

    public static final String MODEL_FILE_POSTFIX = ".model";

    private MnistMatrix[] mnistTrainMatrix;

    private MnistMatrix[] mnistValidationMatrix;

    INDArray trainingInputs = null;

    INDArray trainingOutputs = null;

    INDArray validationInputs = null;

    INDArray validationOutputs = null;

    private MultiLayerConfiguration conf = null;

    private MultiLayerNetwork model = null;

    public static void main(String[] args) {
        System.out.println("MnistSimple");

        int selection = 0;
        if (args.length == 0) {
            System.out.println("No command-line arguments provided.");
            System.out.println("0 for default");
            System.out.println("1 for net initialization with zeros");
            System.exit(1);
        } else {
            selection = Integer.parseInt(args[0]);
        }

        if (selection == 0) { // create and train standard model/net
            MnistSimple mnistSimple = new MnistSimple();
            mnistSimple.createTrainSaveMnistModel();
            return;
        }

        if (selection == 1) { // create and train extended model/net
            MnistSimple mnistSimple = new MnistSimple();
            mnistSimple.createTrainSaveMnistModelInitValuesAreZeros();
            return;
        }

        if (selection == 2) { // just display images
            MnistSimple mnistSimple = new MnistSimple();
            mnistSimple.displayImages();
            return;
        }
    }

    public void createTrainSaveMnistModel() {
        loadData();
        createNet();
        trainNet();
        saveModel();
    }

    public void createTrainSaveMnistModelInitValuesAreZeros() {
        loadData();
        createNet196();
        trainNet();
        saveModel196();
    }

    public void validate() {
        int detectionCounter = 0;
        double sum = 0.0;
        for (int i = 0; i < mnistValidationMatrix.length; i++) {
            INDArray evalInputRow = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
            evalInputRow.putRow(0, validationInputs.getRow(i));
            EvalResult evalResult = askModel(evalInputRow);

            if (i < 2)
                System.out.println("evalResult.label: " + evalResult.label + "  mnistValidateMatrix[" + i + "].getLabel(): " + mnistValidationMatrix[i].getLabel());
            if (evalResult.label == mnistValidationMatrix[i].getLabel()) {
                detectionCounter++;
            }

            sum = sum + evalResult.value;
        }
        System.out.println("mean detection value: " + sum / (mnistValidationMatrix.length * 1.0) + "  detection rate: " + detectionCounter / (mnistValidationMatrix.length * 1.0));
    }

    private void trainNet() {
        // https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/android/first-steps

        // show first two images
        printImage(0);
        printImage(1);

        printEval(0);
        printEval(1);

        double detectionRate = 0;
        DataSet data = new DataSet(trainingInputs, trainingOutputs);
        for (int loop = 0; loop < 1000 || detectionRate < 0.95; loop++) {
            model.fit(data);
            if (loop % 10 == 0) {
                System.out.println("------------------------------------------------------------------");
                System.out.println("Loop: " + loop);

                // eval whole input
                int detectionCounter = 0;
                double sum = 0.0;
                for (int i = 0; i < mnistTrainMatrix.length; i++) {
                    if (i <= 1) {
                        printEval(i);
                    }

                    INDArray evalInputRow = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
                    evalInputRow.putRow(0, trainingInputs.getRow(i));
                    EvalResult evalResult = askModel(evalInputRow);

                    if (i < 2)
                        System.out.println("evalResult.label: " + evalResult.label + "  mnistTrainMatrix[" + i + "].getLabel(): " + mnistTrainMatrix[i].getLabel());
                    if (evalResult.label == mnistTrainMatrix[i].getLabel()) {
                        detectionCounter++;
                    }

                    sum = sum + evalResult.value;
                }
                detectionRate = detectionCounter / (mnistTrainMatrix.length * 1.0);
                System.out.println("mean detection value: " + sum / (mnistTrainMatrix.length * 1.0) + "  detection rate: " + detectionRate);
            }
        }
        System.out.println();
        printEval(0);
        printEval(1);
    }

    public void printEval(int i) {
        INDArray evalInput = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        evalInput.putRow(0, trainingInputs.getRow(i));
        List<INDArray> outList = model.feedForward(evalInput, false);
        INDArray out = outList.get(outList.size() - 1);
        System.out.println(out.toString());
        System.out.println("winner: " + askModel(evalInput).label + " " + askModel(evalInput).value + " label: " + mnistTrainMatrix[i].getLabel());
    }

    public void printImage(INDArray indArray) {
        System.out.println("----------------------------");
        System.out.println("INDArray");
        for (int row = 0; row < mnistTrainMatrix[0].getNumberOfRows(); row++) {
            for (int col = 0; col < mnistTrainMatrix[0].getNumberOfColumns(); col++) {
                int value = indArray.getDouble(0, row * IMG_SIZE + col) > 0 ? 1 : 0;
                System.out.print("" + value);
            }
            System.out.println();
        }
    }

    public void printImage(int i) {
        mnistTrainMatrix[i].print();

        System.out.println("----------------------------");
        System.out.println("INDArray");
        for (int row = 0; row < mnistTrainMatrix[0].getNumberOfRows(); row++) {
            for (int col = 0; col < mnistTrainMatrix[0].getNumberOfColumns(); col++) {
                int value = trainingInputs.getDouble(i, row * IMG_SIZE + col) > 0 ? 1 : 0;
                System.out.print("" + value);
            }
            System.out.println();
        }

        for (int c = 0; c < 10; c++) {
            double value = trainingOutputs.getDouble(i, c);
            System.out.print("" + value + ", ");
        }
        System.out.println();
    }

    public MnistMatrix getTrainingImage(int i) {
        return mnistTrainMatrix[i];
    }

    public EvalResult askModel(INDArray input) {
        List<INDArray> outList = model.feedForward(input, false);
        INDArray out = outList.get(outList.size() - 1);

        int winner = 0;
        double winnerValue = 0;

        for (int i = 0; i < out.columns(); i++) {
            if (out.getDouble(0, i) > winnerValue) {
                winner = i;
                winnerValue = out.getDouble(0, i);
            }
        }

        return new EvalResult(winner, winnerValue);
    }

    public double askModel(INDArray input, int digit) {
        List<INDArray> outList = model.feedForward(input, false);
        INDArray out = outList.get(outList.size() - 1);
        return out.getDouble(0, digit);
    }

    public void createNet() {

        conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(IMG_SIZE * IMG_SIZE)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();

        System.out.println("summary start");
        System.out.println(model.summary());
        System.out.println("summary end");
    }

    public void createNet196() {

        conf = new NeuralNetConfiguration.Builder()
                .seed(4711)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(IMG_SIZE * IMG_SIZE)
                        .nOut(196)
                        .activation(Activation.RELU)
                        //! .weightInit(WeightInit.ONES)
                        //! .weightInit(WeightInit.ZERO)
                        .weightInit(WeightInit.XAVIER)

                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        //! .weightInit(WeightInit.ONES)
                        //! .weightInit(WeightInit.ZERO)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();

        /*
        Layer[] layers = model.getLayers();
        for(int l=0; l<layers.length; l++) {
            Layer layer = layers[l];

            INDArray paramsOfLayer = layer.params();
            long layerSize = layer.numParams();
            System.out.println("layer.numParams(): " + layer.numParams());
            System.out.println("paramsOfLayer.size(0): " + paramsOfLayer.size(0));


            for(int p=0; p<paramsOfLayer.size(0); p++) {
                paramsOfLayer.put(0, p, Math.random());
            }
            // layer.setParams(Nd4j.rand(paramsOfLayer.shape()));
        }
        */

        // INDArray all_params = model.params();
        // model.setParams(Nd4j.rand(all_params.shape()));//set random values with the same shape

        System.out.println("summary start");
        System.out.println(model.summary());
        System.out.println("summary end");
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

    public INDArray getMnistTrainImage(int i) {
        return trainingInputs.getRow(i);
    }

    public void saveModel() {
        try {
            String name = MNIST_SIMPLE_FILE_NAME + MODEL_FILE_POSTFIX;
            ModelSerializer.writeModel(this.model, name, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadModel() {
        try {
            String name = MNIST_SIMPLE_FILE_NAME + MODEL_FILE_POSTFIX;
            model = ModelSerializer.restoreMultiLayerNetwork(name);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveModel196() {
        try {
            String name = MNIST_SIMPLE_FILE_NAME_INIT_VALUES_ARE_ZERO + MODEL_FILE_POSTFIX;
            ModelSerializer.writeModel(this.model, name, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadModel196() {
        try {
            String name = MNIST_SIMPLE_FILE_NAME_INIT_VALUES_ARE_ZERO + MODEL_FILE_POSTFIX;
            model = ModelSerializer.restoreMultiLayerNetwork(name);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void displayImages() {
        loadData();

        for (int i = 0; i < 10; i++) {
            System.out.println("mnistTrainMatrix[" + i + "].getLabel(): " + mnistTrainMatrix[i].getLabel());
        }

        printImage(9);
    }
}

