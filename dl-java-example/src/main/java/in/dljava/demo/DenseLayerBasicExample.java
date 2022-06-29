package in.dljava.demo;

import java.util.List;

import in.dljava.data.Shape;
import in.dljava.functions.loss.LossFunction;
import in.dljava.functions.metrics.MetricsFunction;
import in.dljava.functions.optimizer.AdamOptimizer;
import in.dljava.funtions.activation.ActivationFunction;
import in.dljava.layers.Dense;
import in.dljava.layers.Input;
import in.dljava.models.Sequential;

public class DenseLayerBasicExample {

	public static final void main(String... args) {

		Sequential model = new Sequential();
		model.addLayer(new Input(new Shape(2)));
		model.addLayer(new Dense(2).setActivation(ActivationFunction.RELU));
		model.addLayer(new Dense(1).setActivation(ActivationFunction.SIGMOID));
		model.compile(new AdamOptimizer(), LossFunction.BINARY_CROSSENTROPY, List.of(MetricsFunction.ACCURACY));
	}
}
