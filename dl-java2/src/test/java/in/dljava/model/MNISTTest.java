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

class MNISTTest {

	@Test
	void test() {

		DoubleData xTrain = MNISTReader.readImages("mnist/train-images-idx3-ubyte");
		DoubleData yTrain = MNISTReader.readLabels("mnist/train-labels-idx1-ubyte");

		DoubleData xTest = MNISTReader.readImages("mnist/t10k-images-idx3-ubyte");
		DoubleData yTest = MNISTReader.readLabels("mnist/t10k-labels-idx1-ubyte");

		var optim = new SGDOptimizer(0.0001);

		Sequential model = new Sequential(List.of(
				new DenseLayer(32, new in.dljava.operations.Sigmoid()),
				new DenseLayer(64, new in.dljava.operations.Sigmoid()),
				new DenseLayer(32, new in.dljava.operations.Sigmoid()),
				new DenseLayer(10, new in.dljava.operations.Softmax())), new MeanSquaredError(), 0);

		optim.setModel(model);

		Trainer trainer = new Trainer(model, optim);
		trainer.fit(xTrain, yTrain, xTest, yTest, 100, 10, 32, true);

		for (int i = 0; i < 10; i++) {
			
			int r = new Random().nextInt(0, 5000);
			
			System.out.println("Number : ");
			yTest.subDataNth(r).print();
			System.out.println("Output : ");
			model.forward(xTest.subDataNth(r)).print();
			System.out.println("------");
			
		}
	}
}
