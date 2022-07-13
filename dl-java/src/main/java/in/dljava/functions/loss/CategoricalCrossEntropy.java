package in.dljava.functions.loss;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;

public class CategoricalCrossEntropy implements Loss {

	@Override
	public DoubleData compute(DoubleData output, DoubleData expected) {

		double p = 0;

		for (int i = 0; i < output.getData().length; i++) {
			p += (expected.getData()[i] == 1d ? output.getData()[i] : (1 - output.getData()[i]));
		}

		return new DoubleData(new Shape(1), new double[] { -Math.log(p) });
	}

}
