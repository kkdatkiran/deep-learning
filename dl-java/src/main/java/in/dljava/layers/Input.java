package in.dljava.layers;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.functions.optimizer.OptimizerFunction;
import in.dljava.util.StringUtil;
import in.dljava.util.Tuple2;
import in.dljava.util.Tuples;

public class Input implements Layer {

	private final Shape shape;
	private String name;
	private DoubleData output;

	public Input(Shape shape) {

		this.shape = shape;
	}

	public Shape getShape() {

		return this.shape;
	}

	@Override
	public void compile(String name, Layer previous) {

		this.name = name;
	}

	@Override
	public int getUnits() {

		return shape.total();
	}

	@Override
	public String summary() {

		return String.format("%30s%30s%d", StringUtil.padEnding("Input (" + this.name + ")", 30),
				StringUtil.padEnding(this.shape.toString(), 30), 0);
	}

	@Override
	public Tuple2<Integer, Integer> parameters() {

		return Tuples.of(0, 0);
	}

	@Override
	public DoubleData feedForward(DoubleData prevLayerData) {

		this.output = prevLayerData;
		return this.output;
	}
	
	@Override
	public void backwardPropagation(DoubleData exp) {
		
	}

	@Override
	public DoubleData getOutput() {

		return this.output;
	}

	@Override
	public void print() {
		System.out.println("Input (" + this.name + ") - no paramters to train");
	}

	@Override
	public void updateWeights(DoubleData doubleData, OptimizerFunction optimizer) {

	}

	@Override
	public DoubleData getErrors() {
		return null;
	}
}
