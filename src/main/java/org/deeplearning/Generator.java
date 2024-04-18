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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;
import java.util.Random;

import static org.deeplearning.MnistData.IMG_SIZE;

public class Generator {

    public static final int RANDOM_SIZE = 16;

    public static final int FIRST_LAYER_OUTPUT_SIZE = 512;

    public static final int SECOND_LAYER_OUTPUTSIZE = 1024;

    private MultiLayerConfiguration conf = null;
    private MultiLayerNetwork model = null;
    Random r = null;
    private RandomInput randomInput = null;

    public Generator() {
        createNet();

        randomInput = new RandomInput(model);
        randomInput.generateRandomizedInput();
    }



    public void createNet() {

        conf = new NeuralNetConfiguration.Builder()
                .seed(12345678)
                .weightInit(WeightInit.XAVIER)
                // .biasInit(0.0)
                //.updater(Updater.ADAM)
                .updater(new Adam(0.00005))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(RANDOM_SIZE)
                        .nOut(FIRST_LAYER_OUTPUT_SIZE)
                        // .activation(Activation.RELU)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(FIRST_LAYER_OUTPUT_SIZE)
                        .nOut(SECOND_LAYER_OUTPUTSIZE)
                        // .activation(Activation.RELU)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(SECOND_LAYER_OUTPUTSIZE)
                        .nOut(IMG_SIZE*IMG_SIZE)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())

                .build();

        model = new MultiLayerNetwork(conf);
        model.init();

        int countLayers = model.getnLayers();
        for(int l=0; l<countLayers; l++) {
            System.out.println("layer " + l + " number of weights " + model.getLayer(l).numParams());
        }

    }

    public INDArray generateImage() {
        randomInput.generateRandomizedInput();
        List<INDArray> outList = model.feedForward(randomInput.getRandomInput(), false);
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
        System.out.println("train generator with successful fake");

        DataSet data = new DataSet(randomInput.getRandomInput().reshape(1, RANDOM_SIZE), generatedImage.reshape(1, IMG_SIZE * IMG_SIZE));
        model.fit(data);
    }


    public void trainSuccessfulFake(INDArray image, INDArray randomInputParameter) {
        System.out.println("train generator with successful fake");

        DataSet data = new DataSet(randomInputParameter.reshape(1, RANDOM_SIZE), image.reshape(1, IMG_SIZE * IMG_SIZE));
        model.fit(data);
    }


    public int getHashIndex(int countImages) {
        double sum = 0;
        for(int i=0; i<randomInput.getRandomInput().columns(); i++) {
            sum = sum + randomInput.getRandomInput().getDouble(0, i);
        }
        sum = (sum * countImages) / randomInput.getRandomInput().columns();
        int index = (int) sum;

        return index % countImages;
    }

    public void trainGeneratorWithTrueImage(MnistData mnistData, int digit) {
        // System.out.println("trainGeneratorWithTrueImage");

        int countImages = mnistData.getTrainingInputs().rows();

        DataSet data = new DataSet(randomInput.getRandomInput().reshape(1, RANDOM_SIZE), mnistData.getIndexImage(getHashIndex(countImages), digit).reshape(1, IMG_SIZE * IMG_SIZE));
        // DataSet data = new DataSet(randomInput.reshape(1, RANDOM_SIZE), mnistData.getRandomImage(digit).reshape(1, IMG_SIZE * IMG_SIZE));
        model.fit(data);
    }

    public INDArray getRandomInputLayer() {
        return randomInput.getRandomInput();
    }


    public RandomInput getRandomInput() {
        return randomInput;
    }

    public void  setRandomINDArray(INDArray input) {
        for(int c=0; c<randomInput.getRandomInput().columns(); c++) {
            double value = input.getDouble(0, c);
            randomInput.getRandomInput().put(0, c, value);
        }
    }

    public INDArray manipulateRandomWeightAndRollbackIfNecessary(INDArray originImage, Discriminator discriminator, double discriminatorScore) {
        int layers = model.getnLayers();
        int layerNumber = new Random().nextInt(layers);

        String parameter = "" + layerNumber + "_W";

        INDArray w = model.getParam(parameter);

        int row = new Random().nextInt(w.rows());
        int col = new Random().nextInt(w.columns());

        double originWeight = w.getDouble(row, col);

        double newWeight = new Random().nextDouble();

        w.put(row, col, newWeight);

        INDArray newImage = generateImage(randomInput.getRandomInput());

        double discriminatorNewScore = discriminator.getScore(newImage);


        if(discriminatorNewScore < discriminatorScore) { // rollback
            //! System.out.println("discriminatorScore   : " + discriminatorScore);
            w.put(row, col, originWeight);
            return originImage;
        }
        else {
            //! System.out.println("discriminatorNewScore: " + discriminatorNewScore);
            return newImage;
        }
    }


    public void manipulateWeightsToZero() {
        this.randomInput.manipulateWeightsToZero();
    }

    public void manipulateWeightsToOne() {
        this.randomInput.manipulateWeightsToOne();
    }

    public void manipulateWeightsToRandom() {
        this.randomInput.manipulateWeightsToRandom();
    }

    public void displayWeights() {
        int layerNumber = 0;

        String parameter = "" + layerNumber + "_W";

        INDArray w = model.getParam(parameter);
        System.out.println("display weights" + layerNumber);
        System.out.println(w);
    }
}
