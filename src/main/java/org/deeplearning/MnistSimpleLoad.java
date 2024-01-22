package org.deeplearning;

public class MnistSimpleLoad {

    public static void main(String[] args) {
        System.out.println("useSavedModel");

        MnistSimple mnistSimple = new MnistSimple();
        mnistSimple.createNet();
        mnistSimple.loadModel();
        mnistSimple.loadData();
        mnistSimple.validate();
    }
}
