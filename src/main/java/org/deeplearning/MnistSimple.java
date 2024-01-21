package org.deeplearning;

import org.mnistDataReader.MnistDataReader;
import org.mnistDataReader.MnistMatrix;

import java.io.IOException;

public class MnistSimple {
    MnistMatrix[] mnistTrainMatrix;
    MnistMatrix[] mnistValidateMatrix;

    public static void main(String[] args) {

        System.out.println("MnistSimple");

        MnistSimple mnistSimple = new MnistSimple();

        mnistSimple.loadData();
        mnistSimple.createNet();
        // mnistSimple.validate();

    }

    private void createNet() {
/*
MultiLayerConfiguration cfg = new NeuralNetConfiguration.Builder()
  .weightInit(WeightInit.UNIFORM)
  .list()
  .layer(0,new DenseLayer.Builder()
    .activation(Activation.SIGMOID)
    .nIn(2)
    .nOut(3)
    .build())
  .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
    .activation(Activation.SIGMOID)
    .nIn(3)
    .nOut(1)
    .build())
  .build();
MultiLayerNetwork network = new MultiLayerNetwork(cfg);
network.init();
 */

/*
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(2)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SIGMOID)
                        .nOut(1).build())
                .build();


MultiLayerNetwork model = new MultiLayerNetwork(conf);
model.init();
 */
    }

    private void loadData() {
        try {
            mnistTrainMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
            mnistValidateMatrix = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
