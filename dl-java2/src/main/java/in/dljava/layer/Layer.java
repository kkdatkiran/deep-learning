package in.dljava.layer;

import lombok.Data;

@Data
public abstract class Layer {

	protected int units;
	protected double[] output;
	protected double[] theta;
	protected double[][] delta;
	protected double[][] weight;
}
