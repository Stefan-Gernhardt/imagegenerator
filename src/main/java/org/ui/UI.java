package org.ui;


import main.GenerateImage;

import java.util.List;
import java.util.concurrent.TimeUnit;


public class UI {
	public static final int f = 4;
	public static final int TIME_OUT = 100000;

	private GameUI gameUI = null;

	public UI(boolean withUI) {
		gameUI = new GameUI(withUI);
	}


	public void mainLoop(GenerateImage generateImage) {
		gameUI.start();

		int timeoutTicks = 0;
		do {
			timeoutTicks++;
			gameUI.draw(generateImage);
			// generateImage.findBetterImage();
			generateImage.findBetterImage();

		} while((gameUI.getGameState() != GameUI.Stopped_We_have_a_winner) && (timeoutTicks<TIME_OUT));

	}

	public void mainLoop(List<GenerateImage> generatedImages) {
		gameUI.start();

		int timeoutTicks = 0;
		do {
			timeoutTicks++;
			gameUI.draw(generatedImages);
			for(int i=0; i< GameUI.rows*GameUI.cols; i++) {
				generatedImages.get(i).findBetterImage();
			}
		} while((gameUI.getGameState() != GameUI.Stopped_We_have_a_winner) && (timeoutTicks<TIME_OUT));
	}


	public void wait_(long ms) {
		try {
			TimeUnit.MILLISECONDS.sleep(ms);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}


}
