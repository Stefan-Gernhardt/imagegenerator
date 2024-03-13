package main;

import org.deeplearning.Discriminator;
import org.deeplearning.Generator;
import org.deeplearning.MnistSimple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ui.UI;

import java.util.Random;

import static org.deeplearning.MnistSimple.IMG_SIZE;

public class GenerateImage {
    private MnistSimple mnistSimple = null;
    private MnistSimple mnistSimple196 = null;
    Generator generator = null;
    Discriminator discriminator = null;
    INDArray imageINDArray = null;
    private int digit = 0;

    private boolean goodImageCreated = false;
    private int iteration = 0;


    public GenerateImage() {
        iteration = 0;
        goodImageCreated = false;

        imageINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);

        // mnistSimple = new MnistSimple();
        // mnistSimple.createNet();
        // mnistSimple.loadData();
        // mnistSimple.validate();

        // mnistSimple196 = new MnistSimple();
        // mnistSimple196.createNet196();
        // mnistSimple196.loadModel196();

        generator = new Generator();
        discriminator = new Discriminator();

        // Random random = new Random();
        // for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        //    imageINDArray.put(0, i, random.nextDouble());
        //}
    }

    int getDigit() {
        return digit;
    }

    void setDigit(int d) {
        digit =d;
    }

    public void generate(int digit) {
        setDigit(digit);

        INDArray zeroInput = Nd4j.zeros(1, IMG_SIZE*IMG_SIZE);
        double similarity = mnistSimple.askModel(zeroInput, digit);
        System.out.println("similarity zero input to digit " + digit + " is " + similarity);

        Random random = new Random();
        INDArray randomInput = Nd4j.zeros(1, IMG_SIZE*IMG_SIZE);
        for(int i=0; i<IMG_SIZE*IMG_SIZE; i++) {
            randomInput.put(0, i, random.nextDouble());
        }
        similarity = mnistSimple.askModel(randomInput, digit);
        System.out.println("similarity random input to digit " + digit + " is " + similarity);


        INDArray trainImage5Input = Nd4j.zeros(1, IMG_SIZE*IMG_SIZE);
        for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
            trainImage5Input.put(0, i, mnistSimple.getTrainingImage(0).getValue(i/IMG_SIZE, i%IMG_SIZE));
        }
        similarity = mnistSimple.askModel(trainImage5Input, digit);
        System.out.println("similarity trainingImage5 as input to digit " + digit + " is " + similarity);


        UI ui = new UI(true);
        ui.mainLoop(this);
    }

    public void generateWithDiscriminatorAndGenerator() {
        setDigit(5);
        UI ui = new UI(true);
        ui.mainLoop(this);
    }

    public double getGrayValue(int row, int col) {
        return imageINDArray.getDouble(0, row*IMG_SIZE+col);
    }

    public void findBetterImage() {
        if(goodImageCreated) return;

        iteration++;
        System.out.println("iteration: " + iteration);

        double similarityBefore = mnistSimple.askModel(imageINDArray, digit);
        double similarity196Before = mnistSimple196.askModel(imageINDArray, digit);
        double minSimilarityBefore = Math.min(similarityBefore, similarity196Before);

        Random random = new Random();
        int index = random.nextInt(IMG_SIZE*IMG_SIZE);

        double beforeValue = imageINDArray.getDouble(0, index);
        imageINDArray.put(0, index, random.nextDouble());

        double similarityAfter = mnistSimple.askModel(imageINDArray, digit);
        double similarity196After = mnistSimple196.askModel(imageINDArray, digit);
        double minSimilarityAfter = Math.min(similarityAfter, similarity196After);

        if(minSimilarityAfter <= minSimilarityBefore) {
            imageINDArray.put(0, index, beforeValue);
            System.out.println("similarity:    " + similarityBefore);
        }
        else {
            System.out.println("similarity:    " + similarityAfter);
        }

        double similarity196 = mnistSimple196.askModel(imageINDArray, digit);
        System.out.println("similarity196: " + similarity196);

        if(similarityAfter >= 0.999) {
            this.mnistSimple.printImage(0);
            this.mnistSimple.printImage(imageINDArray);
            goodImageCreated = true;
        }

    }

    public void findBetterImageForDiscriminatorAndGenerator() {
        imageINDArray = generator.askModel();

    }
}
