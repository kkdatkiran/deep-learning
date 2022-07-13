package in.dljava.model;

import org.junit.jupiter.api.Test;

import in.dljava.activation.Relu;
import in.dljava.activation.Sigmoid;
import in.dljava.layer.DenseLayer;
import in.dljava.layer.InputLayer;

class SequentialTest {

	@Test
	void test() {
		Sequential model = new Sequential();
		model.addLayer(new InputLayer(2));
		model.addLayer(new DenseLayer(2).setActivation(new Relu()));
		model.addLayer(new DenseLayer(1).setActivation(new Sigmoid()));
		
		double[][] x = new double[][] {
			new double[] {0d, 0d},
			new double[] {0d, 1d},
			new double[] {1d, 0d},
			new double[] {1d, 1d},
		};
		
		double[][] y = new double[][] {
			new double[] {0d},
			new double[] {1d},
			new double[] {1d},
			new double[] {0d},
		};
		
//		model.fit(x, y, 4, 10, true);
//		
//		System.out.println(model.predict(x));
//		
//		model.print();
	}

}
