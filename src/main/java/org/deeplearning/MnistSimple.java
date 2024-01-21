package org.deeplearning;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.mnistDataReader.MnistDataReader;
import org.mnistDataReader.MnistMatrix;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.List;

// Sources:
// https://gist.github.com/tomthetrainer/7cb2fbc14a5c631a567a98c3134f7dd6
// https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/android/image-classification
// https://medium.com/mlearning-ai/neural-networks-getting-started-with-eclipse-deeplearning4j-897f3662832b



public class MnistSimple {
    private MnistMatrix[] mnistTrainMatrix;
    private MnistMatrix[] mnistValidateMatrix;
    private MultiLayerConfiguration conf = null;
    private MultiLayerNetwork model = null;


    public static void main(String[] args) {

        System.out.println("MnistSimple");

        MnistSimple mnistSimple = new MnistSimple();
        mnistSimple.createTrainSaveMnistModel();
    }

    public void createTrainSaveMnistModel() {
        loadData();
        createNet();
        trainNet();
        // validate();

    }

    private void trainNet() {
        // https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/android/first-steps

        INDArray trainingInputs = Nd4j.zeros(1, 28*28);
        INDArray trainingOutputs = Nd4j.zeros(1, 10);

        for(int r=0; r<28; r++) {
            for(int c=0; c<28; c++) {
                double d = mnistTrainMatrix[0].getValue(r, c) / 255.0;
                trainingInputs.put(0, r*28 + c, d);
            }
        }

        trainingOutputs.put(0, mnistTrainMatrix[0].getLabel(), 1.0 );


        DataSet data = new DataSet(trainingInputs, trainingOutputs);

        List<INDArray> outList = model.feedForward(trainingInputs, false);
        System.out.println("outlist size: " + outList.size());
        INDArray out = outList.get(2);
        System.out.println(out.shapeInfoToString());
        System.out.println(out.toString());

        System.out.println("winner: " + askModel(trainingInputs));

        for(int loop=0; loop<100; loop++) {
            model.fit(data);
        }

        System.out.println("winner: " + askModel(trainingInputs));
    }


    public int askModel(INDArray input) {
        List<INDArray> outList = model.feedForward(input, false);
        INDArray out = outList.get(2);

        int winner = 0;
        double winnerValue = 0;

        for(int i=0; i< out.columns(); i++) {
            if(out.getDouble(0, i) > winnerValue) {
                winner = i;
                winnerValue = out.getDouble(0, i);
            }
        }

        return winner;
    }

    private void createNet() {

        conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(28 * 28)
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

        /*
        //define the layers of the network
        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(4)
                .nOut(3)
                .name("Input")
                .build();

        DenseLayer hiddenLayer = new DenseLayer.Builder()
                .nIn(3)
                .nOut(3)
                .name("Hidden")
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(3)
                .nOut(3)
                .name("Output")
                .activation(Activation.SOFTMAX)
                .build();
         */


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
