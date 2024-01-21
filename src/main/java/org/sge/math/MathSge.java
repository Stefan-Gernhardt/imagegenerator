package org.sge.math;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MathSge {
	
	public static String convertDecTo(int base, int number) {
		if(number == 0) return "0";
		
		String returnValue = "";
		int ganzZahlQuotient = 0;

		do {
			ganzZahlQuotient = number / base;
			int rest = number % base;
			returnValue = "" + rest + returnValue;
			number = ganzZahlQuotient;
		} while(number > 0);
			
		
		return returnValue;
	}

	
	public static String convertDecTo(int base, int number, int countDigits) {
		String returnValue = "";
		int ganzZahlQuotient = 0;

		do {
			ganzZahlQuotient = number / base;
			int rest = number % base;
			returnValue = "" + rest + returnValue;
			number = ganzZahlQuotient;
		} while(number > 0);

		while (returnValue.length() < countDigits) {
			returnValue = "0" + returnValue;
		}
		
		return returnValue;
	}
	
	
	public static int convertStringTo(int base, String number) {
		int sum = 0;
		int power = 1;
		for(int i=0; i<number.length(); i++) {
			int digit = Integer.parseInt(number.substring(number.length()-i-1, number.length()-i));
			sum = sum + (digit * power);
			power = power * base;
		}
		
		return sum;
	}
	
	
	private static Random sgeRandom = null;
	
	public static Random getSgeRandom() {
		if(sgeRandom == null) {
			sgeRandom = new Random();
		}
		
		return sgeRandom;
	}
	
	public static void setSgeRandomSeed(long seed) {
		sgeRandom = new Random(seed);
	}

	
	public static boolean compareINDArrayDim1xN(INDArray a1, INDArray a2, double eps) {
		if(!a1.equalShapes(a2)) throw new RuntimeException("unequalShapes");
		
		for(int i=0; i<a1.columns(); i++) {
			double d = a1.getDouble(0, i) - a2.getDouble(0, i);
			if( Math.abs(d) > eps) {
				return false;
			}
		}
		return true;
	}
	
	
	public static boolean compareINDArrayDim1xNwithDimN(INDArray a1, INDArray a2, double eps) {
		for(int i=0; i<a1.columns(); i++) {
			double d = a1.getDouble(0, i) - a2.getDouble(i);
			if( Math.abs(d) > eps) {
				return false;
			}
		}
		return true;
	}
}
