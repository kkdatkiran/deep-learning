package in.dljava.functions.loss;

import in.dljava.data.DoubleData;

public interface Loss {

	DoubleData compute(DoubleData output, DoubleData expected);
}
