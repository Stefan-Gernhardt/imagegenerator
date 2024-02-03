package org.ui;


import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.image.BufferStrategy;

import javax.swing.JFrame;

public class WindowUI {
	private JFrame frame = null;
	private GameUI game = null;
	
	/**
	 * Create the frame.
	 * 
	 * @param title - desired title of the game
	 * @param game  - the game
	 */
	public WindowUI(String title, GameUI game) {
		frame = new JFrame(title);
		this.game = game;


		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setResizable(true); //!!
		frame.add(game); // Game inherits from Canvas, a Component object, so it can be put in a JFrame
		frame.pack();
		frame.setLocationRelativeTo(null); // ghetto way of centering the window
		frame.setVisible(true);
	}

	
	int getWidth() {
		return frame.getWidth();
	}
	
	int getHeigth() {
		return frame.getHeight();
	}
	
	
	int getWidthCanvas() {
		return frame.getWidth();
	}
	
	int getHeigthCanvas() {
		return frame.getHeight();
	}
	

}