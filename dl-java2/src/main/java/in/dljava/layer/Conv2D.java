package in.dljava.layer;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.functions.initializers.InitializerFunction;
import in.dljava.operations.Conv2DOperation;
import in.dljava.operations.Dropout;
import in.dljava.operations.Flatten;
import in.dljava.operations.Operation;

public class Conv2D extends Layer {

	private int paramSize;
	private double dropout = 1;
	private boolean flatten = false;
	private Operation activation;

	public Conv2D(int neurons, int paramSize, double dropout, Operation activation, boolean flatten) {

		super(neurons);
		this.paramSize = paramSize;
		this.activation = activation;
		this.flatten = flatten;
		this.dropout = dropout;
	}

	@Override
	public void setupLayer(DoubleData numIn) {
		
		var initializer = InitializerFunction.GLOROT_UNIFORM.make(Double.class);
		
		var convParam = (DoubleData) initializer.initalize(new Shape(numIn.getShape().dimensions()[1], this.neurons, this.paramSize, this.paramSize));
		
		this.params.add(convParam);
		this.operations.add(new Conv2DOperation(convParam));
		this.operations.add(this.activation);
		
		if (this.flatten)
			this.operations.add(new Flatten());
		
		if (this.dropout < 1.0d)
			this.operations.add(new Dropout(this.dropout));
	}

	@Override
	public Conv2D deepCopy() {
		
		Conv2D conv = new Conv2D(this.neurons, paramSize, dropout, activation, flatten);
		
		conv.first = this.first;
		conv.params = this.params == null ? null : this.params.stream().map(DoubleData::deepCopy).toList();
		conv.paramGrads = this.paramGrads == null ? null : this.paramGrads.stream().map(DoubleData::deepCopy).toList();
		conv.operations = this.operations == null ? null : this.operations.stream().map(Operation::deepCopy).toList();
		conv.seed = this.seed;

		conv.input = this.input.deepCopy();
		conv.output = this.output.deepCopy();
		
		conv.paramSize = this.paramSize;
		conv.dropout = this.dropout;
		conv.flatten = this.flatten;
		conv.activation = this.activation;
		
		return conv;
	}

	public int getParamSize() {
		return paramSize;
	}

	public double getDropout() {
		return dropout;
	}

	public boolean isFlatten() {
		return flatten;
	}

	public Operation getActivation() {
		return activation;
	}

}
