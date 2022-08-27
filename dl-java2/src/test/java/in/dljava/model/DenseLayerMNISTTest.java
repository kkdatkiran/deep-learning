package in.dljava.model;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import in.dljava.data.DoubleData;
import in.dljava.file.MNISTReader;
import in.dljava.layer.DenseLayer;
import in.dljava.loss.MeanSquaredError;
import in.dljava.optimizer.SGDOptimizer;
import in.dljava.trainer.Trainer;

class DenseLayerMNISTTest {

	@Test
	void test() {

		DoubleData xTrain = MNISTReader.readImages("mnist/train-images-idx3-ubyte").reShape(60000, 28 * 28);
		DoubleData yTrain = MNISTReader.readLabels("mnist/train-labels-idx1-ubyte");

		DoubleData xTest = MNISTReader.readImages("mnist/t10k-images-idx3-ubyte").reShape(10000, 28 * 28);
		DoubleData yTest = MNISTReader.readLabels("mnist/t10k-labels-idx1-ubyte");

		var optim = new SGDOptimizer(0.1);

		Sequential model = new Sequential(List.of(new DenseLayer(32, new in.dljava.operations.Sigmoid()),
				new DenseLayer(64, new in.dljava.operations.Sigmoid()),
				new DenseLayer(32, new in.dljava.operations.Sigmoid()),
				new DenseLayer(10, new in.dljava.operations.Sigmoid())), new MeanSquaredError(), 0);

		optim.setModel(model);

		System.out.println("");
		Trainer trainer = new Trainer(model, optim);
		long start = System.currentTimeMillis();
		trainer.fit(xTrain, yTrain, xTest, yTest, 100, 10, 32, true);
		System.out.println("\n" + ((System.currentTimeMillis() - start) / (60d * 1000)) + " minutes");
		System.out.println("");

		for (int i = 0; i < 10; i++) {

			int r = new Random().nextInt(0, 5000);

			var y = yTest.subDataNth(r);
			y.print();
			System.out.println("Number : " + (y.indexMax()));

			var predy = model.forward(xTest.subDataNth(r), true);
			predy.print();
			System.out.println("Predicted Number : " + (predy.indexMax()));
			System.out.println("------\n\n");

		}
	}
}
