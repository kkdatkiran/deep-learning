package in.dljava.demo;

import in.dljava.activation.Activation;
import in.dljava.data.Shape;
import in.dljava.layers.Dense;
import in.dljava.layers.Input;
import in.dljava.models.Sequential;

public class DenseLayerBasicExample {

	public static final void main(String... args) {

		Sequential model = new Sequential();
		model.addLayer(new Input(new Shape(2)));
		model.addLayer(new Dense(16).setActivation(Activation.SIGMOID));
		model.addLayer(new Dense(1).setActivation(Activation.SIGMOID));
		model.compile()
	}
}
