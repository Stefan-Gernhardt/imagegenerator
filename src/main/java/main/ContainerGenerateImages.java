package main;

import org.ui.GameUI;
import org.ui.UI;

import java.util.ArrayList;
import java.util.List;

public class ContainerGenerateImages {
    private ArrayList<GenerateImage> generatedImages = null;
    private UI ui = null;

    public ContainerGenerateImages() {
        generatedImages = new ArrayList<GenerateImage>();
        for(int i = 0; i< GameUI.rows*GameUI.cols; i++) {
            GenerateImage gi = new GenerateImage();
            gi.setUpGenerateWithDiscriminatorAndGenerator();
            gi.setDigit(i);
            generatedImages.add(gi);
        }

        ui = new UI(true);
    }

    public void run() {
        ui.mainLoop(generatedImages);
    }

}

