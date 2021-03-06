package in.dljava.model;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.file.CSVFileReader;
import in.dljava.layer.DenseLayer;
import in.dljava.loss.MeanSquaredError;
import in.dljava.optimizer.SGDOptimizer;
import in.dljava.optimizer.Tuple4;
import in.dljava.trainer.Trainer;
import in.dljava.util.DataSplit;

class SpiralDataTest {

	@Test
	void test() {

		var lines = CSVFileReader.readCSVFile("spiraldata/spiraldata.csv");

		double x[] = new double[lines.size() * 2];
		double y[] = new double[lines.size()];

		int i = 0;
		for (List<String> line : lines) {

			x[(i * 2)] = Double.valueOf(line.get(0));
			x[(i * 2) + 1] = Double.valueOf(line.get(1));
			y[i++] = Double.valueOf(line.get(2));
		}

		DoubleData xData = new DoubleData(new Shape(lines.size(), 2), x);
		DoubleData yData = new DoubleData(new Shape(lines.size(), 1), y);

		Tuple4<DoubleData, DoubleData, DoubleData, DoubleData> data = DataSplit.trainTestSplit(xData, yData, 0.3);

		var optim = new SGDOptimizer(0.01);

		Sequential model = new Sequential(List.of(
				new DenseLayer(200, new in.dljava.operations.Sigmoid()),
				new DenseLayer(1, new in.dljava.operations.Sigmoid())), new MeanSquaredError(), 0);

		optim.setModel(model);

		Trainer trainer = new Trainer(model, optim);
		trainer.fit(data.getT1(), data.getT2(), data.getT3(), data.getT4(), 20000, 2000, 32, true);
		
		int testSize = data.getT3().getShape().dimensions()[0];

		for (i = 0; i < 10; i++) {

			int r = new Random().nextInt(0, testSize);

			System.out.println("Expected : ");
			data.getT4().subDataNth(r).print();
			System.out.println("Predicted : ");
			model.forward(data.getT3().subDataNth(r)).print();
			System.out.println("----");

		}
	}
}
