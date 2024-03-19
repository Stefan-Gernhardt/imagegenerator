package org.deeplearning;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.mnistDataReader.MnistMatrix;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

import static org.deeplearning.MnistSimple.IMG_SIZE;

public class Discriminator {
    public final static int IS_TRUTH = 0;
    public final static int IS_FAKE = 1;


    private MultiLayerConfiguration conf = null;
    private MultiLayerNetwork model = null;

    public Discriminator() {
        createNet();
    }

    public void createNet() {
        conf = new NeuralNetConfiguration.Builder()
                .seed(3)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(IMG_SIZE * IMG_SIZE)
                        .nOut(1024)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(1024)
                        .nOut(512)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(512)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                //.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(256)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    public EvalResult askModel(INDArray input, boolean print) {
        List<INDArray> outList = model.feedForward(input, false);
        INDArray out = outList.get(outList.size() - 1);

        int winner = -1;
        double winnerValue = -1.0;

        for (int i = 0; i < out.columns(); i++) {
            if(print) System.out.print(out.getDouble(0, i) + ", ");
            if (out.getDouble(0, i) > winnerValue) {
                winner = i;
                winnerValue = out.getDouble(0, i);
            }
        }

        // System.out.println("winner value: " + winnerValue);

        if(print) System.out.println();

        return new EvalResult(winner, winnerValue);
    }

    public INDArray askModelGetGradient(INDArray input) {
        List<INDArray> outList = model.feedForward(input, false);
        return outList.get(outList.size() - 1);
    }

    public void train(MnistMatrix mnistTrainMatrix, INDArray image) {
        INDArray trainingOutput = Nd4j.zeros(1, 2);
        trainingOutput.put(0,IS_TRUTH, 1.0);
        trainingOutput.put(0, IS_FAKE, askModelGetGradient(image.reshape(1, IMG_SIZE * IMG_SIZE)).getDouble(IS_FAKE));
        System.out.println("" + trainingOutput); //!

        DataSet data = new DataSet(image.reshape(1, IMG_SIZE * IMG_SIZE), trainingOutput);
        model.fit(data);
    }
}
