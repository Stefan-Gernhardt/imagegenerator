package org.deeplearning;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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

import static org.deeplearning.MnistData.IMG_SIZE;

public class Discriminator {
    public final static int INDEX_FOR_PROBABILTY_FOR_REAL_IMAGE = 0;

    private MultiLayerConfiguration conf = null;
    private MultiLayerNetwork model = null;

    public Discriminator() {
        createNet();
    }


    public void createNet() {
        conf = new NeuralNetConfiguration.Builder()
                .seed(3)
                .weightInit(WeightInit.XAVIER)
                //.updater(Updater.ADAM)
                .updater(new Adam(0.00005))
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
                // .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                // .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR) //! try out
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.XENT) // XENT: Cross Entropy: Binary Classification - LossBinaryXENT
                        // We set the loss function to binary_crossentropy,
                        // which is appropriate for binary classification tasks.
                        // However, in multi-class classification,
                        // it can still be used
                        // if each class is treated as a binary classification problem independently.

                        .nIn(256)
                        .nOut(1)
                        //.activation(Activation.SOFTMAX)
                        .activation(Activation.SIGMOID)
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


    public void trainDiscriminatorWithFakeImage(INDArray image) {
        System.out.println("trainDiscriminatorWithFakeImage"); //!
        INDArray trainingOutput = Nd4j.zeros(1, 1);
        trainingOutput.put(0, INDEX_FOR_PROBABILTY_FOR_REAL_IMAGE, 0.0);

        DataSet data = new DataSet(image.reshape(1, IMG_SIZE * IMG_SIZE), trainingOutput);
        model.fit(data);
    }

    public boolean isTheGeneratedImageGoodEnough(INDArray generatedImage) {
        List<INDArray> outList = model.feedForward(generatedImage, false);

        INDArray gradient = outList.get(outList.size() - 1);

        return gradient.getDouble(0, INDEX_FOR_PROBABILTY_FOR_REAL_IMAGE) >= 0.5;
    }

    public INDArray trainDiscriminatorWithTrueImage(MnistData mnistData, int digit) {
        System.out.println("trainDiscriminatorWithTrueImage"); //!
        INDArray trainingOutput = Nd4j.zeros(1, 1);
        trainingOutput.put(0, INDEX_FOR_PROBABILTY_FOR_REAL_IMAGE, 1.0);

        INDArray img = mnistData.getRandomImage(digit);
        DataSet data = new DataSet(img.reshape(1, IMG_SIZE * IMG_SIZE), trainingOutput);
        model.fit(data);

        return img;
    }
}
