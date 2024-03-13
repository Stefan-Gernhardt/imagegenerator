package org.ui;


import main.GenerateImage;

import java.awt.BasicStroke;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Stroke;
import java.awt.image.BufferStrategy;
import java.util.Random;

import static org.deeplearning.MnistSimple.IMG_SIZE;

public class GameUI extends Canvas  {
	
	public final static int Running = 1;
	public final static int Stopped_We_have_a_winner = 2;
	public final static int Reset = 0;
	public final static int OFFSET_X = 16;
	public final static int OFFSET_Y = 39;
	
	private int gameState = Reset;

	private static final long serialVersionUID = 1L;

	public final static int WIDTH  = 200; 
	public final static int HEIGHT = 140; 

	private MainMenu menu = null;
	private static WindowUI windowUI = null;
	
	public GameUI(boolean withUI) {

		if(withUI) {
			canvasSetup();
			windowUI = new WindowUI("Image Generation", this);
		}

		this.addKeyListener(new KeyInput());
		this.addMouseListener(menu);
		this.addMouseMotionListener(menu);
		this.setFocusable(true);		
	}


	public void reset(double startPointVariationDefault, double startAngleVariationDefault) {
		gameState = Reset;
	}
	
	
	public void start() {
		gameState = Running;
	}


	private void canvasSetup() {
		this.setPreferredSize(new Dimension(WIDTH, HEIGHT));
		this.setMinimumSize(new Dimension(WIDTH, HEIGHT));
	}


	public void draw(GenerateImage generateImage) {
		// Initialize drawing tools first before drawing

		BufferStrategy buffer = this.getBufferStrategy(); // extract buffer so we can use them
		// a buffer is basically like a blank canvas we can draw on

		if (buffer == null) { // if it does not exist, we can't draw! So create it please
			this.createBufferStrategy(3); // Creating a Triple Buffer
			/*
			 * triple buffering basically means we have 3 different canvases this is used to
			 * improve performance but the drawbacks are the more buffers, the more memory
			 * needed so if you get like a memory error or something, put 2 instead of 3.
			 * 
			 * BufferStrategy:
			 * https://docs.oracle.com/javase/7/docs/api/java/awt/image/BufferStrategy.html
			 */

			return;
		}

		Graphics g = buffer.getDrawGraphics(); // extract drawing tool from the buffers

		drawBackground(g);
		drawImageArray(g, generateImage);

		g.dispose();
		buffer.show();

	}

	/**
	 * draw the background
	 * 
	 * @param g - tool to draw
	 */
	private void drawBackground(Graphics g) {
		g.setColor(Color.black);
		g.fillRect(0, 0, getw(), geth());
	}


	public void drawImageArray(Graphics g, GenerateImage generateImage) {
		int h = geth();
		int w = getw();

		int hSize = h / IMG_SIZE;
		if(hSize <= 0) hSize = 1;

		int wSize = w / IMG_SIZE;
		if(wSize <= 0) wSize = 1;

		Random random = new Random();
		for(int row = 0; row<IMG_SIZE; row++) {
			for(int col = 0; col<IMG_SIZE; col++) {
				float grayValue = (float) generateImage.getGrayValue(row, col);
				int colorNumber = Color.HSBtoRGB(0, 0, grayValue);
				// int colorNumber = Color.HSBtoRGB(0, 0, random.nextFloat());
				g.setColor(new Color(colorNumber));
				g.fillRect(col*wSize, row*hSize, wSize, hSize);
			}
		}
	}

	/**
	 * update settings and move all objects
	 */
	public void update() {
		// update ball (movements)
		// gameState = Stopped_We_have_a_winner;
	}

	/**
	 * used to keep the value between the min and max
	 * 
	 * @param value - integer of the value we have
	 * @param min   - minimum integer
	 * @param max   - maximum integer
	 * @return: the value if value is between minimum and max, minimum is returned
	 *          if value is smaller than minimum, maximum is returned if value is
	 *          bigger than maximum
	 */
	public static int ensureRange(int value, int min, int max) {
		return Math.min(Math.max(value, min), max);
	}

	/**
	 * returns the sign (either 1 or -1) of the input
	 * 
	 * @param d - a double for the input
	 * @return 1 or -1
	 */
	public static int sign(double d) {
		if (d <= 0)
			return -1;

		return 1;
	}
	
	
	public int getGameState() {
		return this.gameState;
	}



	public static int geth() {
		if(windowUI == null) return HEIGHT; 
		else return windowUI.getHeigth() - GameUI.OFFSET_Y;
	}
	
	
	public static int getw() {
		if(windowUI == null) return WIDTH; 
		else return windowUI.getWidth() - GameUI.OFFSET_X;
	}

}
