package in.dljava.functions.metrics;

import in.dljava.data.DoubleData;

public interface MetricsFunction {

	double compute(DoubleData output, DoubleData expected);
}
