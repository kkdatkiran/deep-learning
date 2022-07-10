package in.dljava.functions.loss;

import in.dljava.DLException;
import in.dljava.data.DoubleData;

public class BinaryCrossEntropy implements Loss {

	@Override
	public double compute(DoubleData output, DoubleData expected) {

		if (output.getShape().total() != 1 || expected.getShape().total() != 1)
			throw new DLException("Binary cross entropy can be calculated if the size is 1");

		double p = expected.getData()[0] == 1d ? output.getData()[0] : (1 - output.getData()[0]);

		return -Math.log(p);
	}

}
