package org.deeplearning;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.Random;

import static org.deeplearning.MnistData.IMG_SIZE;

public class Generator {

    public static final int RANDOM_SIZE = 256;

    private MultiLayerConfiguration conf = null;
    private MultiLayerNetwork model = null;
    Random r = null;
    private INDArray randomInput = null;

    public Generator() {
        randomInput = Nd4j.zeros(1, RANDOM_SIZE);
        r = new Random(0);
        generateRandomizedInput();

        createNet();
    }

    public INDArray generateRandomizedInput() {
        for(int pixel=0; pixel<RANDOM_SIZE; pixel++) {
            randomInput.put(0, pixel, r.nextDouble());
        }
        return randomInput;
    }

    public void createNet() {

        conf = new NeuralNetConfiguration.Builder()
                .seed(12345678)
                .weightInit(WeightInit.XAVIER)
                //.updater(Updater.ADAM)
                .updater(new Adam(0.00005))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(RANDOM_SIZE)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(512)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(1024)
                        .nOut(784)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())

                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    public INDArray generateImage() {
        generateRandomizedInput();
        List<INDArray> outList = model.feedForward(randomInput, false);
        INDArray out = outList.get(outList.size() - 1);
        return out;
    }

    public INDArray generateImage(INDArray input) {
        // System.out.println("" + input);
        List<INDArray> outList = model.feedForward(input, false);
        INDArray out = outList.get(outList.size() - 1);
        return out;
    }

    public void trainSuccessfulFake(INDArray generatedImage) {
        System.out.println("trainSuccessfulFake");

        DataSet data = new DataSet(randomInput.reshape(1, RANDOM_SIZE), generatedImage.reshape(1, IMG_SIZE * IMG_SIZE));
        model.fit(data);
    }


    public int getHashIndex(int countImages) {
        double sum = 0;
        for(int i=0; i<randomInput.columns(); i++) {
            sum = sum + randomInput.getDouble(0, i);
        }
        sum = (sum * countImages) / randomInput.columns();
        int index = (int) sum;

        return index % countImages;
    }

    public void trainGeneratorWithTrueImage(MnistData mnistData, int digit) {
        System.out.println("trainGeneratorWithTrueImage");

        int countImages = mnistData.getTrainingInputs().rows();

        DataSet data = new DataSet(randomInput.reshape(1, RANDOM_SIZE), mnistData.getIndexImage(getHashIndex(countImages), digit).reshape(1, IMG_SIZE * IMG_SIZE));
        // DataSet data = new DataSet(randomInput.reshape(1, RANDOM_SIZE), mnistData.getRandomImage(digit).reshape(1, IMG_SIZE * IMG_SIZE));
        model.fit(data);
    }

    public INDArray getRandomINDArray() {
        return randomInput;
    }
}
