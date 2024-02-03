package org.ui;


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


	public void mainLoop() {
		gameUI.start();

		//! gameUI.draw();

		int timeoutTicks = 0;
		do {
			timeoutTicks++;
			gameUI.draw();

			//! wait_(100);
		} while((gameUI.getGameState() != GameUI.Stopped_We_have_a_winner) && (timeoutTicks<TIME_OUT));

		//! wait_(1000);

	}

	public void playOnePoint() {
		gameUI.start();

		// game loop for one point
		gameUI.draw();

		int gameTicks = 0;
		do {
			gameTicks++;
			gameUI.update();
			gameUI.draw();

			wait_(100);
		} while((gameUI.getGameState() != GameUI.Stopped_We_have_a_winner) && (gameTicks<TIME_OUT));

		System.out.println();
		wait_(1000);

		if(gameTicks >= TIME_OUT) {
			return;
		}
	}


	public void wait_(long ms) {
		try {
			TimeUnit.MILLISECONDS.sleep(ms);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}


}
