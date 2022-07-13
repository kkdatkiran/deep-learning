package in.dljava.models;

import java.util.List;

import org.junit.jupiter.api.Test;

import in.dljava.data.Data;
import in.dljava.data.Shape;
import in.dljava.functions.loss.BinaryCrossEntropy;
import in.dljava.functions.metrics.Accuracy;
import in.dljava.functions.optimizer.MomentumOptimizer;
import in.dljava.funtions.activation.ActivationFunction;
import in.dljava.layers.Dense;
import in.dljava.layers.Input;

class SequentialTest {

	@Test
	void test() {
		Sequential model = new Sequential();
		model.addLayer(new Input(new Shape(2)));
		model.addLayer(new Dense(2).setActivation(ActivationFunction.RELU.make()));
		model.addLayer(new Dense(1).setActivation(ActivationFunction.SIGMOID.make()));
		model.compile(new MomentumOptimizer(0.001), new BinaryCrossEntropy(), List.of(new Accuracy()));
		
		System.out.println(model.summary());
		
		Data x = Data.of(new double[][] {
			new double[] {0d, 0d},
			new double[] {0d, 1d},
			new double[] {1d, 0d},
			new double[] {1d, 1d},
		});
		
		Data y = Data.of(new double[][] {
			new double[] {0d},
			new double[] {1d},
			new double[] {1d},
			new double[] {0d},
		});
		
		model.fit(x, y, 4, 10, true);
		
		System.out.println(model.predict(x));
		
		model.print();
	}

}
