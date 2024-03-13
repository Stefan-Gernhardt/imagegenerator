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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

import static org.deeplearning.MnistSimple.IMG_SIZE;

public class Generator {
    private MultiLayerConfiguration conf = null;
    private MultiLayerNetwork model = null;
    private INDArray standardInput = null;

    public Generator() {
        standardInput = Nd4j.zeros(1, 1);
        standardInput.put(0, 0, 1.0);

        createNet();
    }

    public void createNet() {

        conf = new NeuralNetConfiguration.Builder()
                .seed(12345678)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(1)
                        .nOut(100)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(100)
                        .nOut(IMG_SIZE * IMG_SIZE)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    public INDArray askModel() {
        List<INDArray> outList = model.feedForward(standardInput, false);
        return outList.get(outList.size()-1);
    }

    public INDArray askModelAndRandom() {
        List<INDArray> outList = model.feedForward(standardInput, false);
        INDArray out = outList.get(outList.size()-1);
        return addRandom(out);
    }

    public INDArray addRandom(INDArray input) {
        return input;
    }

}
