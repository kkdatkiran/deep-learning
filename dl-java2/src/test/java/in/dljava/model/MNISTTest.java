package in.dljava.model;

import java.util.List;

import org.junit.jupiter.api.Test;

import in.dljava.data.DoubleData;
import in.dljava.file.MNISTReader;
import in.dljava.layer.DenseLayer;
import in.dljava.loss.MeanSquaredError;
import in.dljava.optimizer.SGDOptimizer;

class MNISTTest {

	@Test
	void test() {
		
		DoubleData xTrain = MNISTReader.readImages("mnist/train-images-idx3-ubyte");
		DoubleData yTrain = MNISTReader.readLabels("mnist/train-labels-idx1-ubyte");
		
		DoubleData xTest = MNISTReader.readImages("mnist/t10k-images-idx3-ubyte");
		DoubleData yTest = MNISTReader.readLabels("mnist/t10k-labels-idx1-ubyte");


		var optim = new SGDOptimizer(0.0001);
//
//		Sequential model = new Sequential(List.of(new DenseLayer(6, new in.dljava.operations.Relu()),
//
//				new DenseLayer(1, new in.dljava.operations.Sigmoid())), new MeanSquaredError(), 0);
//
//		optim.setModel(model);	
	}
}
