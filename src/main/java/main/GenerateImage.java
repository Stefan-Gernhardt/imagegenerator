package main;

import org.deeplearning.Discriminator;
import org.deeplearning.Generator;
import org.deeplearning.MnistSimple;
import org.deeplearning.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ui.UI;

import java.util.Random;

import static org.deeplearning.MnistSimple.IMG_SIZE;

public class GenerateImage {
    private int mode = 0;
    private MnistSimple mnistSimple = null;
    private MnistSimple mnistSimple196 = null;
    Generator generator = null;
    Discriminator discriminator = null;

    public final static int N_GEN = 100;
    public final static int ELITE_N_GEN = 40;
    INDArray geneticINDArray = null;
    INDArray geneticMeanINDArray = null;
    INDArray geneticSTDINDArray = null;
    double [] geneticScore = null;
    INDArray eliteGeneticINDArray = null;

    INDArray imageINDArray = null;
    private int digit = 0;

    private boolean goodImageCreated = false;
    private int iteration = 0;


    public GenerateImage() {
        iteration = 0;
        goodImageCreated = false;

        imageINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);

    }

    private void setUpGenerateSimple() {
        mode = 1;

        mnistSimple = new MnistSimple();
        mnistSimple.createNet();
        mnistSimple.loadData();
        mnistSimple.validate();

        mnistSimple196 = new MnistSimple();
        mnistSimple196.createNet196();
        mnistSimple196.loadModel196();

        // Random random = new Random();
        // for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        //    imageINDArray.put(0, i, random.nextDouble());
        //}
    }

    public void generateSimple() {
        setUpGenerateSimple();
        setDigit(5);
        UI ui = new UI(true);
        ui.mainLoop(this);
    }

    private void setUpGenerateWithDiscriminatorAndGenerator() {
        mode = 2;

        generator = new Generator();
        discriminator = new Discriminator();
    }

    public void generateWithDiscriminatorAndGenerator() {
        setUpGenerateWithDiscriminatorAndGenerator();
        setDigit(5);
        UI ui = new UI(true);
        ui.mainLoop(this);
    }

    private void setUpGenerateWithGeneticAlgo() {
        mode = 3;

        mnistSimple = new MnistSimple();
        mnistSimple.createNet();
        mnistSimple.loadData();

        geneticINDArray = Nd4j.zeros(N_GEN, IMG_SIZE * IMG_SIZE);

        geneticMeanINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        Util.setIndArrayToConstant(geneticMeanINDArray, 0.5);

        geneticSTDINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        Util.setIndArrayToConstant(geneticSTDINDArray, 0.5);

        eliteGeneticINDArray = Nd4j.zeros(ELITE_N_GEN, IMG_SIZE * IMG_SIZE);
        geneticScore = new double[N_GEN];
    }

    public void generateWithGeneticAlgo() {
        setUpGenerateWithGeneticAlgo();
        setDigit(5);

        UI ui = new UI(true);
        ui.mainLoop(this);
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

    public double getGrayValue(int row, int col) {
        return imageINDArray.getDouble(0, row*IMG_SIZE+col);
    }

    public void findBetterImage() {
        if(mode == 1) {
            findBetterImageSimple();
            return;
        }
        if(mode == 2) {
            findBetterImageForDiscriminatorAndGenerator();
            return;
        }
        if(mode == 3) {
            findBetterImageGeneticAlgo();
            return;
        }

        System.out.println("mode not found " + mode);
        System.exit(1);
    }

    public void findBetterImageSimple() {
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

    private void findBetterImageGeneticAlgo() {
        if(iteration >= 2500) return;

        iteration++;
        Util.setIndArrayWithGaussDistribution(geneticINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);

        for(int i=0; i<N_GEN; i++) {
            geneticScore[i] = mnistSimple.askModel(geneticINDArray.getRow(i).reshape(1, IMG_SIZE*IMG_SIZE), digit);
            // System.out.println("score: " + i + " " + geneticScore[i]);
        }

        // System.out.println("geneticINDArray");
        // printOutRowCol(geneticINDArray);

        // System.out.println("elite array with rowcol");
        // printOutRowCol(eliteGeneticINDArray);

        double[] forTest = { 0.036745186895132065, 0.05668370798230171, 0.06619622558355331, 0.0216363538056612, 0.07507462054491043, 0.0508420467376709, 0.1080874502658844, 0.057289399206638336, 0.044289011508226395, 0.07861603051424026 }; //!

        int eliteRow = 0;
        for(int i=0; i<N_GEN; i++) {
            //! if(isEliteScore(forTest[i], forTest)) {
            if(isEliteScore(geneticScore[i], geneticScore)) {
                if(eliteRow < ELITE_N_GEN) {
                    // eliteGeneticINDArray.putRow(eliteRow, geneticINDArray.getRow(i));
                    for(int c=0; c<geneticINDArray.columns(); c++) {
                        double value = geneticINDArray.getDouble(i, c);
                        eliteGeneticINDArray.put(eliteRow, c, value);
                    }
                    eliteRow++;
                }
            }
        }

        // System.out.println("elite array with rowcol");
        // printOutRowCol(eliteGeneticINDArray);

        // System.out.println("geneticMeanINDArray");
        // printOutRowCol(geneticMeanINDArray);

        // System.out.println("geneticSTDINDArray");
        // printOutRowCol(geneticSTDINDArray);

        Util.computeMeanAndStd(eliteGeneticINDArray, geneticMeanINDArray, geneticSTDINDArray);
        System.out.println("geneticMeanINDArray");
        printOutRowCol(geneticMeanINDArray);

        System.out.println("geneticSTDINDArray");
        printOutRowCol(geneticSTDINDArray);

        // Util.setIndArrayWithGaussDistribution(imageINDArray, 0.5, 0.5, 0.0, 1.0); //! pd
        Util.setIndArrayWithGaussDistribution(imageINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);

        double score = mnistSimple.askModel(imageINDArray, digit);
        System.out.println("iteration: " + iteration + "  score: " + score);

        // if(iteration == 25) System.exit(1);
    }

    private boolean isEliteScore(double v, double[] geneticScore) {
        int rank = 1;
        for(int i=0; i<N_GEN; i++) {
            if(v < geneticScore[i]) rank++;
        }

        boolean isElite = rank <= ELITE_N_GEN;
        // if(isElite) {
        //     System.out.println("elite: " + v);
        // }

        return isElite;
    }

    private void printOutRowCol(INDArray indArray) {
        for (int r = 0; r < indArray.rows(); r++) {
            for (int c = 0; c < indArray.columns(); c++) {
                System.out.print("" + indArray.getDouble(r, c) + ",");
            }
            System.out.println();
        }
    }


}
