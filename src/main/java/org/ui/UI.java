package org.ui;


import main.GenerateImage;

import java.util.concurrent.TimeUnit;


public class UI {
	public static final int f = 4;
	public static final int TIME_OUT = 100000;

	public static final int MOVE_UP   = 0;
	public static final int MOVE_DOWN = 1;
	public static final int MOVE_STAY = 2;
	
	private GameUI gameUI = null;

	public static final double[] startPositions = { 0.0, 1.0 };
	public static final double[] startVels =      { 0.0, 0.0 };

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
			generateImage.findBetterImageForDiscriminatorAndGenerator();

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
