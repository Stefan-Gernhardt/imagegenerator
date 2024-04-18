package org.deeplearning;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.deeplearning.Generator.RANDOM_SIZE;

public class RandomInput {

    public final static int DENSE = 4;
    private INDArray randomInput = null;
    private INDArray randomLayer0 = null;
    private INDArray randomLayer1 = null;
    private MultiLayerNetwork model = null;

    private Random rG = null;

    RandomInput(MultiLayerNetwork model) {
        this.model = model;

        rG = new Random(0);

        int layerSize0 = model.layerSize(0);
        // int layerSize1 = model.layerSize(1);
        // int layerSize2 = model.layerSize(2);


        randomInput = Nd4j.zeros(1, RANDOM_SIZE);
        randomLayer0 = Nd4j.zeros(1, model.layerSize(0));
        randomLayer1 = Nd4j.zeros(1, model.layerSize(1));

    }

    public INDArray generateRandomizedInput() {
        for(int pixel=0; pixel<RANDOM_SIZE; pixel++) {
            randomInput.put(0, pixel, rG.nextDouble());
        }
        return randomInput;
    }

    public INDArray getRandomInput() {
        return randomInput;
    }

    public void manipulateWeights(int layerNumber, boolean randomUsed, double value) {
        String parameter = "" + layerNumber + "_W";

        INDArray w = model.getParam(parameter);


        for(int r=0; r<w.rows(); r++) {
            for(int c=0; c<w.columns(); c++) {
                if(c % DENSE == 0) {
                    if (randomUsed) {
                        double v = rG.nextGaussian();
                        w.put(r, c, v);
                    } else {
                        w.put(r, c, value);
                    }
                }
            }
        }

        // System.out.println("weights after maninpulation" + layerNumber);
        // System.out.println(w);

        /*
        String parameterBias = "" + layerNumber + "_b";
        INDArray bias = model.getParam(parameterBias);

        for(int r=0; r<bias.rows(); r++) {
            for(int c=0; c<bias.columns(); c++) {
                if(randomUsed) {
                    double v = rG.nextGaussian();
                    bias.put(r, c, v);
                }
                else {
                    bias.put(r, c, value);
                }
            }
        }
        */

        // System.out.println("weights after maninpulation" + layerNumber);
        // System.out.println(w);

    }

    public void manipulateWeightsToZero() {
        manipulateWeights(0, false, 0.0);
        manipulateWeights(1, false, 0.0);
        manipulateWeights(2, false, 0.0);
    }

    public void manipulateWeightsToOne() {
        manipulateWeights(0, false, 1.0);
        manipulateWeights(1, false, 1.0);
        manipulateWeights(2, false, 1.0);
    }

    public void manipulateWeightsToRandom() {
        manipulateWeights(0, true, -1);
        // manipulateWeights(1, true, -1);
        // manipulateWeights(2, true, -1);
    }
}
