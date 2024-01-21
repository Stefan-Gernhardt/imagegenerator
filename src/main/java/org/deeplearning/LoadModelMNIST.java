package org.deeplearning;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;

import static org.deeplearning.LeNetMNIST.MODEL_PATH;

public class LoadModelMNIST {
    private static final Logger log = LoggerFactory.getLogger(LoadModelMNIST.class);

    public static void main(String[] args) throws Exception {

        log.info("Load model from tmp folder: " + MODEL_PATH);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "lenetmnist.zip");
        MultiLayerNetwork loadedModel = MultiLayerNetwork.load(new File(path), true);



        Layer[] layers = loadedModel.getLayers();
        int countLayers = layers.length;
        System.out.println("countLayers: " + countLayers);

        loadedModel.getLayerNames();
        // loadedModel.feedForward();

        for(Layer l : layers) {
            System.out.println("layer index: " + l.getIndex());
            // System.out.println("layer: " + l.toString());
            System.out.println("layer numParams: " + l.numParams());
            // System.out.println("layer input length: " + l.input().length());
            System.out.println("layer input length: " + l.conf());

        }

        String summary = loadedModel.summary();
        System.out.println(summary);


        // List<INDArray> output = loadedModel.feedForward(); // INDArray input

        /*
        for(INDArray ia : output) {
            System.out.println(ia);
        }

         */
    }

}
