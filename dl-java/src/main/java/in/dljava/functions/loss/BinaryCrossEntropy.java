package in.dljava.functions.loss;

import in.dljava.DLException;
import in.dljava.data.DoubleData;
import in.dljava.data.Shape;

public class BinaryCrossEntropy implements Loss {

	@Override
	public DoubleData compute(DoubleData output, DoubleData expected) {

		if (output.getShape().total() != 1 || expected.getShape().total() != 1)
			throw new DLException("Binary cross entropy can be calculated if the size is 1");

		double p = expected.getData()[0] == 1d ? output.getData()[0] : (1 - output.getData()[0]);

		return new DoubleData(new Shape(1), new double[] { -Math.log(p) });
	}

}
