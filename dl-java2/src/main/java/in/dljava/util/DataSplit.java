package in.dljava.util;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import in.dljava.DLException;
import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.optimizer.Tuple4;

public class DataSplit {

	private DataSplit() {

	}

	public static Tuple4<DoubleData, DoubleData, DoubleData, DoubleData> trainTestSplit(DoubleData xData,
			DoubleData yData, double ratio) {

		int size = xData.getShape().dimensions()[0];

		if (size != yData.getShape().dimensions()[0]) {
			throw new DLException("x data and y data are of not same size");
		}

		int trainSize = (int) Math.floor(size * (1 - ratio));
		int testSize = size - trainSize;

		Set<Integer> finishedIndexes = new HashSet<>();

		int eachXSize = xData.getShape().oneOfHigherDimension().total();
		int eachYSize = yData.getShape().oneOfHigherDimension().total();

		double[] xD = new double[trainSize * eachXSize];
		double[] yD = new double[trainSize * eachYSize];

		Random rnd = new Random();
		for (int i = 0; i < trainSize; i++) {

			int x;
			do {
				x = rnd.nextInt(size);
			} while (finishedIndexes.contains(x));
			finishedIndexes.add(x);

			DoubleData sample = xData.subDataNth(x);
			System.arraycopy(sample.getData(), 0, xD, i * eachXSize, eachXSize);
			sample = yData.subDataNth(x);
			System.arraycopy(sample.getData(), 0, yD, i * eachYSize, eachYSize);
		}

		Tuple4<DoubleData, DoubleData, DoubleData, DoubleData> ret = new Tuple4<>();

		int[] dims = xData.getShape().dimensions();
		dims[0] = trainSize;
		ret.setT1(new DoubleData(new Shape(dims), xD));

		dims = yData.getShape().dimensions();
		dims[0] = trainSize;
		ret.setT2(new DoubleData(new Shape(dims), yD));

		xD = new double[testSize * eachXSize];
		yD = new double[testSize * eachYSize];

		for (int i = 0, j = 0; i < size; i++) {

			if (finishedIndexes.contains(i))
				continue;

			DoubleData sample = xData.subDataNth(i);
			System.arraycopy(sample.getData(), 0, xD, j * eachXSize, eachXSize);
			sample = yData.subDataNth(i);
			System.arraycopy(sample.getData(), 0, yD, j * eachYSize, eachYSize);

			j++;
		}

		dims = xData.getShape().dimensions();
		dims[0] = testSize;
		ret.setT3(new DoubleData(new Shape(dims), xD));

		dims = yData.getShape().dimensions();
		dims[0] = testSize;
		ret.setT4(new DoubleData(new Shape(dims), yD));

		return ret;
	}
}
