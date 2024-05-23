package main;

public class Main {
	

	public static void main(String[] args) {
		System.out.println("Image Generator");
		// new GenerateImage().generateSimple();
		// new GenerateImage().generateWithGeneticAlgo();
		// new GenerateImage().generateWithDiscriminatorAndGenerator();
		// new ContainerGenerateImages().run();
		new ContainerGenerateImagesDenoising().run();
	}

}
