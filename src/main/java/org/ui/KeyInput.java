package org.ui;


import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

/**
 * process key input from the keyboard
 * 
 * @author Zayed
 *
 */
public class KeyInput extends KeyAdapter {

	private boolean lup = false; // lup = left up (up1 in video)
	private boolean ldown = false;

	private boolean rup = false;
	private boolean rdown = false;

	@Override
	public void keyPressed(KeyEvent e) {
		int key = e.getKeyCode();

		if (key == KeyEvent.VK_UP) {
			rup = true;
		}
		if (key == KeyEvent.VK_DOWN) {
			rdown = true;
		}
		if (key == KeyEvent.VK_W) {
			lup = true;
		}
		if (key == KeyEvent.VK_S) {
			ldown = true;
		}

		// exit
		if (key == KeyEvent.VK_ESCAPE) {
			System.exit(1);
		}
	}

	@Override
	public void keyReleased(KeyEvent e) {
		int key = e.getKeyCode();

		if (key == KeyEvent.VK_UP) {
			// rp.stop();
			rup = false;
		}
		if (key == KeyEvent.VK_DOWN) {
			// rp.stop();
			rdown = false;
		}
		if (key == KeyEvent.VK_W) {
			// lp.stop();
			lup = false;
		}
		if (key == KeyEvent.VK_S) {
			// lp.stop();
			ldown = false;
		}

		// this is the magic that will stop the lag
		if (!lup && !ldown)
			// lp.stop();
			;
		if (!rup && !rdown)
			//rp.stop();
			;

	}

}
