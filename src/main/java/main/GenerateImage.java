package main;

import org.deeplearning.Discriminator;
import org.deeplearning.Generator;
import org.deeplearning.MnistData;
import org.deeplearning.MnistSimple;
import org.deeplearning.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ui.GameUI;
import org.ui.UI;

import java.util.ArrayList;
import java.util.Random;

import static org.deeplearning.MnistData.IMG_SIZE;

public class GenerateImage {

    private static final int MEAN_LOOP = 10;

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
    private double globalScore = 0.0;

    private Random random = null;


    public GenerateImage() {
        iteration = 0;
        goodImageCreated = false;

        imageINDArray = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);
        random = new Random(0);
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
        Util.setIndArrayToConstant(geneticMeanINDArray, 0.5 );

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

    public void setUpGenerateWithDiscriminatorAndGenerator() {
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
            Util.copyOneDimIndArrayToTwoDimIndArray(generator.getRandomInput().generateRandomizedInput(), geneticINDArray, i);
        }
    }

    public void generateWithDiscriminatorAndGenerator() {
        setUpGenerateWithDiscriminatorAndGenerator();
        setDigit(5);
        UI ui = new UI(true);

        ArrayList<GenerateImage> images = new ArrayList<GenerateImage>();
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


    public void findBetterImageForDiscriminatorAndGenerator1() {
        if(generateImageOrTakeImage) {
            INDArray generatedImage = generator.generateImage();

            if(discriminator.isTheGeneratedImageGoodEnough(generatedImage, 0.5)) {
                System.out.println("---------------------------------------------");
                System.out.println("iteration: " + iteration++);
                System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + globalScore);
                discriminator.trainDiscriminatorWithFakeImage(generatedImage);
                generator.trainSuccessfulFake(generatedImage);
                imageINDArray = generatedImage;
            }
            else {
                generator.trainGeneratorWithTrueImage(mnistData, digit);
            }
            generateImageOrTakeImage = false;
        }
        else {
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }
    }


    public void findBetterImageForDiscriminatorAndGenerator2() {
        if(generateImageOrTakeImage) {
            INDArray generatedSingleImage = generator.generateImage();

            if(discriminator.isTheGeneratedImageGoodEnough(generatedSingleImage, 0.5)) {
                generateImageOrTakeImage = false;

                System.out.println("---------------------------------------------");
                System.out.println("iteration: " + iteration++);
                double generatedSingleImageScoreMnist = mnistSimple.askModel(generatedSingleImage, digit);
                double generatedSingleImageScore = discriminator.getScore(generatedSingleImage);
                System.out.println("generatedSingleImage score: " + generatedSingleImageScore + " MNistSimple: " + generatedSingleImageScoreMnist + "  " + generator.getRandomInputLayer());

                if(generatedSingleImageScoreMnist < 0.9) {
                    imageINDArray = generatedSingleImage;
                    discriminator.trainDiscriminatorWithFakeImage(generatedSingleImage);
                    generator.trainSuccessfulFake(generatedSingleImage);
                }
                else {
                    INDArray geneticImage = geneticImage(generatedSingleImageScoreMnist);
                    double geneticImageScoreMnist = mnistSimple.askModel(geneticImage, digit);
                    System.out.println("geneticImageScoreMnist: " + geneticImageScoreMnist);
                    discriminator.trainDiscriminatorWithFakeImage(geneticImage);
                    generator.trainSuccessfulFake(geneticImage);
                }
            }
            else {
                generator.trainGeneratorWithTrueImage(mnistData, digit);
            }
        }
        else {
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }
    }

    private INDArray geneticImage(double inputScore) {
        boolean verbose = false;
        INDArray maxImage = Nd4j.zeros(1, IMG_SIZE * IMG_SIZE);

        double maxScoreInit = -1.0;
        int counter=0;
        while(counter<N_GEN) {
            INDArray gI  = generator.generateImage();
            // double score = discriminator.getScore(gI);
            double score =  mnistSimple.askModel(gI, digit);
            if(score > inputScore) {
                Util.copyOneDimIndArrayToTwoDimIndArray(generator.getRandomInputLayer(), geneticINDArray, counter);
                geneticScore[counter] = score;
                counter++;

                if(score > maxScoreInit) {
                    maxScoreInit = score;
                    imageINDArray = gI;
                }
            }
        }

        if(verbose) System.out.println("maxScoreInit: " + maxScoreInit);

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

        if(verbose) {
            for (int i = 0; i < N_GEN; i++) {
                System.out.println("i=" + i + "  " + geneticScore[i] + "  " + geneticINDArray.getRow(i));
            }
        }

        Util.computeMeanAndStd(eliteGeneticINDArray, geneticMeanINDArray, geneticSTDINDArray);
        if(verbose) {
            System.out.println("mean: " + geneticMeanINDArray);
            System.out.println("std: " + geneticSTDINDArray);

            for(int i=0; i<ELITE_N_GEN; i++) {
                System.out.println("i=" + i + "  " + eliteGeneticINDArray.getRow(i));
            }
        }

        double maxScore = -1.0;
        long winnerIndex = -1;
        for(int geneticLoop=0; geneticLoop<MEAN_LOOP; geneticLoop++) {
            maxScore = -1.0;
            winnerIndex = -1;

            int countGeneticLoop = 0;
            while (countGeneticLoop < N_GEN) {
                INDArray distribution = Util.getOneRowWithGaussDistribution(geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);
                INDArray generatedImage = generator.generateImage(distribution);
                // geneticScore[countGeneticLoop] = discriminator.getScore(generatedImage);
                geneticScore[countGeneticLoop] =  mnistSimple.askModel(generatedImage, digit);

                if (geneticScore[countGeneticLoop] > inputScore) {
                    Util.copyOneDimIndArrayToTwoDimIndArray(distribution, geneticINDArray, countGeneticLoop);

                    if(verbose) System.out.println("i=" + countGeneticLoop + "  score: " + geneticScore[countGeneticLoop] + geneticINDArray.getRow(countGeneticLoop));
                    if (geneticScore[countGeneticLoop] >= maxScore) {
                        maxScore = geneticScore[countGeneticLoop];
                        winnerIndex = countGeneticLoop;
                        Util.copyOneDimIndArrayToOneDimIndArray(generatedImage, maxImage);
                    }
                    countGeneticLoop++;
                }
            }
            if(verbose) System.out.println("geneticLoop: " + geneticLoop + " max score: " + maxScore);

            eliteRow = 0;
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

            if(verbose) {
                System.out.println("elite");
                System.out.println(eliteGeneticINDArray);
            }

            Util.computeMeanAndStd(eliteGeneticINDArray, geneticMeanINDArray, geneticSTDINDArray);
            if(verbose) {
                System.out.println("mean: " + geneticMeanINDArray);
                System.out.println("std: " + geneticSTDINDArray);
            }
            Util.setIndArrayWithGaussDistribution(geneticINDArray, geneticMeanINDArray, geneticSTDINDArray, 0.0, 1.0);
        }

        System.out.println("max score: " + maxScore);

        generator.setRandomINDArray(geneticINDArray.getRow(winnerIndex));
        return maxImage;
    }


    public void findBetterImageForDiscriminatorAndGenerator3() {
        INDArray generatedImage = Nd4j.zeros(1, IMG_SIZE*IMG_SIZE);


        if(generateImageOrTakeImage) {
            Util.copyOneDimIndArrayToOneDimIndArray(generator.generateImage(), generatedImage);

            double discriminatorScore = discriminator.getScore(generatedImage);
            System.out.println("---------------------------------------------");
            System.out.println("iteration: " + iteration++);
            System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + globalScore);

            if(discriminatorScore < 0.5) {
                generatedImage = manipulateWeightsOfGenerator(discriminatorScore, generatedImage, discriminator);
                generator.trainGeneratorWithTrueImage(mnistData, digit);
            }

            discriminator.trainDiscriminatorWithFakeImage(generatedImage);
            generator.trainSuccessfulFake(generatedImage);
            imageINDArray = generatedImage;

            generateImageOrTakeImage = false;
        }
        else {
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }
    }


    private INDArray manipulateWeightsOfGenerator(double discriminatorScore, INDArray image, Discriminator discriminator) {
        double score = 0;

        while(discriminatorScore < 0.5) {
            image = generator.manipulateRandomWeightAndRollbackIfNecessary(image, discriminator, discriminatorScore);
            discriminatorScore = discriminator.getScore(image);
            System.out.println("discriminatorScore: " + discriminatorScore);
        }

        return  image;
    }

    public void findBetterImageForDiscriminatorAndGenerator4() {
        if(generateImageOrTakeImage) {
            INDArray generatedImage = generator.generateImage();
            System.out.println("---------------------------------------------");
            System.out.println("generatedImage: " + generatedImage);

            // System.out.println("input layer random: " + generator.getRandomInputLayer());

            double score = mnistSimple.askModel(generatedImage, digit);

            if(discriminator.isTheGeneratedImageGoodEnough(generatedImage, 0.5)) {
                globalScore = score;
                System.out.println("iteration true: " + iteration++);
                System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + score);
                discriminator.trainDiscriminatorWithFakeImage(generatedImage);
                generator.trainSuccessfulFake(generatedImage);
                imageINDArray = generatedImage;
                generateImageOrTakeImage = false;
            }
            else {
                // generator.trainGeneratorWithTrueImage(mnistData, digit);

                System.out.println("iteration false: " + iteration++);
                System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + score);

                generator.manipulateWeightsToRandom();

                /*
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to random: " + discriminator.askModelGetGradient(generatedImage));

                if(discriminator.isTheGeneratedImageGoodEnough(generatedImage, 0.5)) {
                    globalScore = score;
                    System.out.println("iteration true: " + iteration++);
                    System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + score);
                    discriminator.trainDiscriminatorWithFakeImage(generatedImage);
                    generator.trainSuccessfulFake(generatedImage);
                    imageINDArray = generatedImage;
                    generateImageOrTakeImage = false;
                }

                for(int l=0; l<10; l++) {
                    generator.manipulateWeightsToRandom();
                    generatedImage = generator.generateImage(generator.getRandomInputLayer());
                    System.out.println("generatedImage: " + generatedImage);
                    System.out.println("manipulate weights to random: " + discriminator.askModelGetGradient(generatedImage));
                }


                generator.manipulateWeightsToRandom();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to random: " + discriminator.askModelGetGradient(generatedImage));

                generator.manipulateWeightsToRandom();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to random: " + discriminator.askModelGetGradient(generatedImage));

                generator.manipulateWeightsToZero();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to zero  : " + discriminator.askModelGetGradient(generatedImage));


                generator.manipulateWeightsToOne();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to one   : " + discriminator.askModelGetGradient(generatedImage));

                generator.manipulateWeightsToZero();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to one   : " + discriminator.askModelGetGradient(generatedImage));

                generator.manipulateWeightsToOne();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to one   : " + discriminator.askModelGetGradient(generatedImage));

                generator.manipulateWeightsToRandom();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to random: " + discriminator.askModelGetGradient(generatedImage));

                generator.manipulateWeightsToRandom();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to random: " + discriminator.askModelGetGradient(generatedImage));

                generator.manipulateWeightsToRandom();
                generatedImage = generator.generateImage(generator.getRandomInputLayer());
                System.out.println("generatedImage: " + generatedImage);
                System.out.println("manipulate weights to random: " + discriminator.askModelGetGradient(generatedImage));

                // generator.displayWeights();

                System.out.println();
                */
            }
        }
        else {
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }
    }

    public void findBetterImageForDiscriminatorAndGenerator() {
        if(generateImageOrTakeImage) {
            INDArray generatedImage = generator.generateImage();
            double scoreMnistSimple = mnistSimple.askModel(generatedImage, digit);

            if(scoreMnistSimple > globalScore) {
                globalScore = scoreMnistSimple;
                imageINDArray = generatedImage;
            }

            if(discriminator.isTheGeneratedImageGoodEnough(generatedImage, 0.5)) {
                System.out.println("---------------------------------------------");
                System.out.println("iteration: " + iteration++);
                System.out.println("gradient of generated image discriminator: " + discriminator.askModelGetGradient(generatedImage) + "  MnistSimple: " + globalScore);
                discriminator.trainDiscriminatorWithFakeImage(generatedImage);
                generator.trainSuccessfulFake(generatedImage);
            }
            else {
                generator.trainGeneratorWithTrueImage(mnistData, digit);
            }
            generateImageOrTakeImage = false;
        }
        else {
            discriminator.trainDiscriminatorWithTrueImage(mnistData, digit);
            generateImageOrTakeImage = true;
        }
    }




}
