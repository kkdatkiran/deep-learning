package in.dljava.model;

import java.util.List;

import org.junit.jupiter.api.Test;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.layer.DenseLayer;
import in.dljava.loss.MeanSquaredError;
import in.dljava.optimizer.SGDOptimizer;
import in.dljava.trainer.Trainer;

class SequentialTest {

	@Test
	void test() {
		var optim = new SGDOptimizer(0.0001);

		Sequential model = new Sequential(List.of(new DenseLayer(6, new in.dljava.operations.Relu()),
				new DenseLayer(1, new in.dljava.operations.Sigmoid())), new MeanSquaredError(), 0);

		optim.setModel(model);

		double[] x = new double[] { 0, 0, 0, 1, 1, 0, 1, 1 };

		double[] y = new double[] { 0, 1, 1, 0 };

		Trainer trainer = new Trainer(model, optim);
		DoubleData xData = new DoubleData(new Shape(4, 2), x);
		DoubleData yData = new DoubleData(new Shape(4, 1), y);
		trainer.fit(xData, yData, xData, yData, 100, 10, 1, true);
		
		System.out.println(model.forward(new DoubleData(new Shape(1,2), new double[] {1,0})));
	}

}
