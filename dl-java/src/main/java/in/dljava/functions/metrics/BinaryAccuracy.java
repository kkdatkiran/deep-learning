package in.dljava.functions.metrics;

import in.dljava.DLException;
import in.dljava.data.DoubleData;

public class BinaryAccuracy implements MetricsFunction {
	
	private static final double THRESHOLD = 0.5d;

	@Override
	public double compute(DoubleData output, DoubleData expected) {
		
		if (output.getShape().total() != 1 || expected.getShape().total() != 1)
			throw new DLException("Binary accuracy can be calculated if the size is 1");
		
		double[] e = expected.getData();
		double[] o = output.getData();

		return ((o[0] < THRESHOLD ? 0d : 1d) == e[0]) ? 1d : 0d;
	}
	
	@Override
	public String toString() {

		return "Binary Accuracy";
	}

}
