package in.dljava.functions.metrics;

import in.dljava.data.DoubleData;

public class Accuracy implements MetricsFunction {

	private static final double THRESHOLD = 0.5d;

	@Override
	public double compute(DoubleData output, DoubleData expected) {

		double count = 0d;
		double[] e = expected.getData();
		double[] o = output.getData();

		for (int i = 0; i < e.length; i++) {

			count += ((o[i] < THRESHOLD ? 0d : 1d) == e[i]) ? 1d : 0d;
		}

		return count / e.length;
	}
	
	@Override
	public String toString() {

		return "Accuracy";
	}
}
