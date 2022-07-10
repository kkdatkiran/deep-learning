package in.dljava.functions.loss;

import in.dljava.data.DoubleData;

public interface Loss {

	double compute(DoubleData output, DoubleData expected);
}
