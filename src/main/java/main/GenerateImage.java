package main;

import org.deeplearning.Discriminator;
import org.deeplearning.Generator;
import org.deeplearning.MnistData;
import org.deeplearning.MnistSimple;
import org.deeplearning.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ui.UI;

import java.util.Random;

import static org.deeplearning.MnistData.IMG_SIZE;

public class GenerateImage {
    private static final int MEAN_LOOP = 50;

    private int mode = 0;

    private MnistSimple mnistSimple = null;

    private MnistSimple mnistSimple196 = null;

    Generator generator = null;

    Discriminator discriminator = null;

    public final static int N_GEN = 10;

    public final static int ELITE_N_GEN = 4;

    INDArray geneticINDArray = null;

    INDArray geneticMeanINDArray = null;
    INDArray saveGeneticMeanINDArray = null;

    INDArray geneticSTDINDArray = null;
    INDArray saveGeneticSTDINDArray = null;

    double[] geneticScore = null;

    INDArray eliteGeneticINDArray = null;

    INDArray imageINDArray = null;

    private int digit = 0;

    private boolean goodImageCreated = false;

    private boolean generateImageOrTakeImage = true;
    MnistData mnistData = null;
    private int iteration = 0;


    public GenerateImage() {
        iteration = 0;
        goodImageCreated = false;

        imageINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
    }

    public void loadMnistSimple() {
        mnistSimple = new MnistSimple();
        mnistSimple.createNet();
        mnistSimple.loadModel();
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

    private void setUpGenerateWithGeneticAlgo() {
        mode = 3;

        mnistSimple = new MnistSimple();
        mnistSimple.createNet();
        mnistSimple.loadData();

        geneticINDArray = Nd4j.zeros(N_GEN, IMG_SIZE * IMG_SIZE);

        geneticMeanINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        saveGeneticMeanINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        Util.setIndArrayToConstant(geneticMeanINDArray, 0.5);

        geneticSTDINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        saveGeneticSTDINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
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

    private void setUpGenerateWithDiscriminatorAndGenerator() {
        mode = 2;

        generator = new Generator();
        discriminator = new Discriminator();

        mnistData = new MnistData();

        loadMnistSimple();

        geneticINDArray = Nd4j.zeros(N_GEN, Generator.RANDOM_SIZE);

        geneticMeanINDArray = Nd4j.zeros(1, Generator.RANDOM_SIZE);
        saveGeneticMeanINDArray = Nd4j.zeros(1, Generator.RANDOM_SIZE);
        Util.setIndArrayToConstant(geneticMeanINDArray, 0.5);

        geneticSTDINDArray = Nd4j.zeros(1, Generator.RANDOM_SIZE);
        saveGeneticSTDINDArray = Nd4j.zeros(1, Generator.RANDOM_SIZE);
        Util.setIndArrayToConstant(geneticSTDINDArray, 0.5);

        eliteGeneticINDArray = Nd4j.zeros(ELITE_N_GEN, Generator.RANDOM_SIZE);
        geneticScore = new double[N_GEN];

        for(int i=0; i<N_GEN; i++) {
            Util.copyOneDimIndArrayToTwoDimIndArray(generator.generateRandomizedInput(), geneticINDArray, i);
        }
    }

    public void generateWithDiscriminatorAndGenerator() {
        setUpGenerateWithDiscriminatorAndGenerator();
        setDigit(5);
        UI ui = new UI(true);
        ui.mainLoop(this);
    }

    int getDigit() {
        return digit;
    }

    void setDigit(int d) {
        digit = d;
    }

    public void generate(int digit) {
        setDigit(digit);

        INDArray zeroInput = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        double similarity = mnistSimple.askModel(zeroInput, digit);
        System.out.println("similarity zero input to digit " + digit + " is " + similarity);

        Random random = new Random();
        INDArray randomInput = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
            randomInput.put(0, i, random.nextDouble());
        }
        similarity = mnistSimple.askModel(randomInput, digit);
        System.out.println("similarity random input to digit " + digit + " is " + similarity);

        INDArray trainImage5Input = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        for (int i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
            trainImage5Input.put(0, i, mnistSimple.getTrainingImage(0).getValue(i / IMG_SIZE, i % IMG_SIZE));
        }
        similarity = mnistSimple.askModel(trainImage5Input, digit);
        System.out.println("similarity trainingImage5 as input to digit " + digit + " is " + similarity);

        UI ui = new UI(true);
        ui.mainLoop(this);
    }

    public double getGrayValue(int row, int col) {
        return imageINDArray.getDouble(0, row * IMG_SIZE + col);
    }

    public void findBetterImage() {
        if (mode == 1) {
            findBetterImageSimple();
            return;
        }
        if (mode == 2) {
            findBetterImageForDiscriminatorAndGenerator();
            return;
        }
        if (mode == 3) {
            findBetterImageGeneticAlgo();
            return;
        }

        System.out.println("mode not found " + mode);
        System.exit(1);
    }

    public void findBetterImageSimple() {
        if (goodImageCreated) return;

        iteration++;
        System.out.println("iteration: " + iteration);

        double similarityBefore = mnistSimple.askModel(imageINDArray, digit);
        double similarity196Before = mnistSimple196.askModel(imageINDArray, digit);
        double minSimilarityBefore = Math.min(similarityBefore, similarity196Before);

        Random random = new Random();
        int index = random.nextInt(IMG_SIZE * IMG_SIZE);

        double beforeValue = imageINDArray.getDouble(0, index);
        imageINDArray.put(0, index, random.nextDouble());

        double similarityAfter = mnistSimple.askModel(imageINDArray, digit);
        double similarity196After = mnistSimple196.askModel(imageINDArray, digit);
        double minSimilarityAfter = Math.min(similarityAfter, similarity196After);

        if (minSimilarityAfter <= minSimilarityBefore) {
            imageINDArray.put(0, index, beforeValue);
            System.out.println("similarity:    " + similarityBefore);
        } else {
            System.out.println("similarity:    " + similarityAfter);
        }

        double similarity196 = mnistSimple196.askModel(imageINDArray, digit);
        System.out.println("similarity196: " + similarity196);

        if (similarityAfter >= 0.999) {
            this.mnistSimple.printImage(0);
            this.mnistSimple.printImage(imageINDArray);
            goodImageCreated = true;
        }
    }

    private void findBetterImageGeneticAlgo() {
        iteration++;
        if (iteration >= 2500) return;

        double meanScoreBefore = calculateMeanScore();

        System.out.println("iteration: " + iteration + "  meanScoreBefore: " + meanScoreBefore);
        if (meanScoreBefore > 0.92) return;

        Util.setIndArrayWithGaussDistribution(geneticINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);

        for (int i = 0; i < N_GEN; i++) {
            geneticScore[i] = mnistSimple.askModel(geneticINDArray.getRow(i).reshape(1, IMG_SIZE * IMG_SIZE), digit);
            // System.out.println("score: " + i + " " + geneticScore[i]);
        }

        // System.out.println("geneticINDArray");
        // printOutRowCol(geneticINDArray);

        // System.out.println("elite array with rowcol");
        // printOutRowCol(eliteGeneticINDArray);

        int eliteRow = 0;
        for (int i = 0; i < N_GEN; i++) {
            if (isEliteScore(geneticScore[i], geneticScore)) {
                if (eliteRow < ELITE_N_GEN) {
                    // eliteGeneticINDArray.putRow(eliteRow, geneticINDArray.getRow(i));
                    for (int c = 0; c < geneticINDArray.columns(); c++) {
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

        copyIndArrayElementByElement(geneticMeanINDArray, saveGeneticMeanINDArray);
        copyIndArrayElementByElement(geneticSTDINDArray, saveGeneticSTDINDArray);

        Util.computeMeanAndStd(eliteGeneticINDArray, geneticMeanINDArray, geneticSTDINDArray);
        // System.out.println("geneticMeanINDArray");
        // printOutRowCol(geneticMeanINDArray);

        // System.out.println("geneticSTDINDArray");
        // printOutRowCol(geneticSTDINDArray);

        Util.setIndArrayWithGaussDistribution(imageINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);

        double meanScoreAfter1 = calculateMeanScore();
        System.out.println("iteration: " + iteration + "  meanscore after1: " + meanScoreAfter1);

        double meanScoreAfter2 = calculateMeanScore();
        System.out.println("iteration: " + iteration + "  meanscore after2: " + meanScoreAfter2);

        System.out.println("geneticMeanINDArray changed values");
        printOutRowCol(geneticMeanINDArray);

        if (meanScoreBefore > meanScoreAfter1) {
            copyIndArrayElementByElement(saveGeneticMeanINDArray, geneticMeanINDArray);
            copyIndArrayElementByElement(saveGeneticSTDINDArray, geneticSTDINDArray);

            System.out.println("geneticMeanINDArray restored values");
            printOutRowCol(geneticMeanINDArray);
        }

        double meanScoreAfter3 = calculateMeanScore();
        System.out.println("iteration: " + iteration + "  meanscore after3: " + meanScoreAfter3);
        System.out.println("----------------------------------------");
    }

    private boolean isEliteScore(double v, double[] geneticScore) {
        int rank = 1;
        for (int i = 0; i < N_GEN; i++) {
            if (v < geneticScore[i]) rank++;
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

    private double calculateMeanScore() {
        double meanScore = 0;
        for(int i = 0;i<MEAN_LOOP;i++) {
            Util.setIndArrayWithGaussDistribution(imageINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);
            double score = mnistSimple.askModel(imageINDArray, digit);
            meanScore = meanScore + score;
        }
        return meanScore / MEAN_LOOP;
    }

    public void copyIndArrayElementByElement(INDArray a1, INDArray a2) {
        for (int r = 0; r < a1.rows(); r++) {
            for (int c = 0; c < a1.columns(); c++) {
                double value = a1.getDouble(r, c);
                a2.put(r, c, value);
            }
        }
    }


    public void foundGoodGeneratedImage(INDArray generatedImage) {
        System.out.println("generated image is good enough");
        discriminator.trainDiscriminatorWithFakeImage(generatedImage);
        imageINDArray = generatedImage;
        generateImageOrTakeImage = false;

        generator.trainSuccessfulFake(generatedImage);
    }


    public void findBetterImageForDiscriminatorAndGenerator() {
        System.out.println("---------------------------------------------");
        System.out.println("iteration: " + iteration++);

        if(generateImageOrTakeImage) {
            INDArray generatedImage = generator.generateImage();
            System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + mnistSimple.askModel(generatedImage, digit));

            if(discriminator.isTheGeneratedImageGoodEnough(generatedImage)) {
                foundGoodGeneratedImage(generatedImage);
            }
            else {
                System.out.println("Try up to 10 times to generated a better image");
                int counter = 10;
                while(counter > 0 && generateImageOrTakeImage) {
                    counter--;
                    generatedImage = generator.generateImage();
                    System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + mnistSimple.askModel(generatedImage, digit));
                    if(discriminator.isTheGeneratedImageGoodEnough(generatedImage)) {
                        foundGoodGeneratedImage(generatedImage);
                    }
                }

                if(generateImageOrTakeImage) {
                    System.out.println("No better image found");
                    generator.trainGeneratorWithTrueImage(mnistData, digit);
                    generateImageOrTakeImage = false;
                }
            }
        }
        else {
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }
        if(iteration>10000000) System.exit(1); //!
    }


    public void findBetterImageForDiscriminatorAndGenerator2() {
        // if(iteration >= 1) return;

        System.out.println("---------------------------------------------");
        System.out.println("iteration: " + iteration++);


        if(generateImageOrTakeImage) {
            INDArray maxImage = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
            for(int i=0; i<N_GEN; i++) {
                Util.copyOneDimIndArrayToTwoDimIndArray(generator.generateRandomizedInput(), geneticINDArray, i);
            }

            double maxScore = -1.0;
            do {
                maxScore = -1.0;

                for (int i = 0; i < N_GEN; i++) {
                    INDArray generatedImage = generator.generateImage(geneticINDArray.getRow(i).reshape(1, Generator.RANDOM_SIZE));

                    geneticScore[i] = discriminator.getScore(generatedImage);
                    System.out.println("score: " + geneticScore[i]);
                    if (geneticScore[i] >= maxScore) {
                        maxScore = geneticScore[i];
                        Util.copyOneDimIndArrayToOneDimIndArray(generatedImage, maxImage);
                    }
                }
                System.out.println("max score: " + maxScore);

                int eliteRow = 0;
                for (int i = 0; i < N_GEN; i++) {
                    if (isEliteScore(geneticScore[i], geneticScore)) {
                        if (eliteRow < ELITE_N_GEN) {
                            for (int c = 0; c < geneticINDArray.columns(); c++) {
                                double value = geneticINDArray.getDouble(i, c);
                                eliteGeneticINDArray.put(eliteRow, c, value);
                            }
                            eliteRow++;
                        }
                    }
                }
                Util.computeMeanAndStd(eliteGeneticINDArray, geneticMeanINDArray, geneticSTDINDArray);
                System.out.println("mean: " + geneticMeanINDArray);
                System.out.println("std: " + geneticSTDINDArray);
                Util.setIndArrayWithGaussDistribution(geneticINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);
            } while(maxScore < 0.5);

            // System.out.println("generated image is good enough");
            discriminator.trainDiscriminatorWithFakeImage(maxImage);
            generator.trainSuccessfulFake(maxImage);

            imageINDArray = maxImage;
            generateImageOrTakeImage = false;
        }
        else { // take real image
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }




            /*
            if(generateImageOrTakeImage) {
                System.out.println("try genetic algo to find a good image");
                Util.setIndArrayWithGaussDistribution(geneticINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);

                int maxIndex = 0;
                double maxScore = -1.0;
                for (int i = 0; i < N_GEN; i++) {
                    //!! geneticScore[i] = discriminator.getScore(geneticINDArray.getRow(i).reshape(1, ));
                    System.out.println(geneticINDArray.getRow(i));
                    System.out.println("i: " + i + "  score: " + geneticScore[i]);
                    if(geneticScore[i] >= maxScore) {
                        maxIndex = i;
                        maxScore = geneticScore[i];
                    }
                }

                System.out.println("max score genetic algo: " + maxScore);

                while (maxScore < 0.5) {
                    int eliteRow = 0;
                    for (int i = 0; i < N_GEN; i++) {
                        if (isEliteScore(geneticScore[i], geneticScore)) {
                            if (eliteRow < ELITE_N_GEN) {
                                for (int c = 0; c < geneticINDArray.columns(); c++) {
                                    double value = geneticINDArray.getDouble(i, c);
                                    eliteGeneticINDArray.put(eliteRow, c, value);
                                }
                                eliteRow++;
                            }
                        }
                    }

                    Util.computeMeanAndStd(eliteGeneticINDArray, geneticMeanINDArray, geneticSTDINDArray);
                    Util.setIndArrayWithGaussDistribution(geneticINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);

                    for (int i = 0; i < N_GEN; i++) {
                        geneticScore[i] = discriminator.getScore(generatedImage);
                        if(geneticScore[i] >= maxScore) {
                            System.out.println("i: " + i + "  score: " + geneticScore[i]);
                            maxIndex = i;
                            maxScore = geneticScore[i];
                        }
                    }
                    System.out.println("max score genetic algo: " + maxScore);
                    System.exit(1);
                }

                System.exit(1);
            }
        }
        else {
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }
        */

        // test put image on UI
        // for(int p=0; p<IMG_SIZE*IMG_SIZE; p++) {
        //    imageINDArray.put(0, p, mnistData.getTrainingInputs().getDouble(1, p));
        //}

        // INDArray img = discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
        // imageINDArray = img.reshape(1,IMG_SIZE * IMG_SIZE);


        // System.out.println("validateIsTruthScoreAverage: " + mnistData.validateIsTruthScoreAverage(discriminator, digit));

        // mnistData.trainDiscriminatorWithTrueImage(discriminator, digit);

        // System.exit(1); //!
        if(iteration>10000000) System.exit(1); //!
    }

}
