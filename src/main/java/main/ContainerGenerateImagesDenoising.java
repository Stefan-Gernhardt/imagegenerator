package main;

import org.deeplearning.GeneratorDenoising;
import org.deeplearning.MnistData;
import org.deeplearning.MnistSimple;
import org.ui.GameUI;
import org.ui.UI;

import java.util.ArrayList;

public class ContainerGenerateImagesDenoising {
    private ArrayList<GenerateImage> generatedImages = null;
    private UI ui = null;

    private MnistSimple mnistSimple = null;
    private MnistData mnistData = null;

    private GeneratorDenoising generatorDenoising = null;

    public ContainerGenerateImagesDenoising() {
        generatorDenoising = new GeneratorDenoising();

        mnistData = new MnistData();
        mnistSimple = new MnistSimple();
        mnistSimple.createNet();
        mnistSimple.loadModel();

        generatedImages = new ArrayList<GenerateImage>();
        for(int i = 0; i< GameUI.rows*GameUI.cols; i++) {
            GenerateImage gi = new GenerateImage();
            gi.setUpGenerateWithDenoising(generatorDenoising, mnistSimple, mnistData);
            gi.setDigit(3); //!!
            generatedImages.add(gi);
        }

        ui = new UI(true);
    }

    public void run() {
        ui.mainLoop(generatedImages);
    }

}
