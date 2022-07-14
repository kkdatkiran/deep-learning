package in.dljava.layer;

import java.util.ArrayList;
import java.util.List;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.functions.initializers.InitializerFunction;
import in.dljava.operations.BiasAdd;
import in.dljava.operations.Operation;
import in.dljava.operations.WeightMultiply;

public class DenseLayer extends Layer {

	private Operation activation;

	public DenseLayer(int neurons, Operation activation) {

		super(neurons);
		this.activation = activation;
	}

	@Override
	public void setupLayer(DoubleData input) {

		this.params = new ArrayList<>();

		var initializer = InitializerFunction.GLOROT_UNIFORM.make(Double.class);

		// Weights
		this.params.add((DoubleData) initializer.initalize(new Shape(input.getShape().dimensions()[1], this.neurons)));

		initializer = InitializerFunction.ZEROS.make(Double.class);
		// Bias
		this.params.add((DoubleData) initializer.initalize(new Shape(1, this.neurons)));

		this.operations = List.of(new WeightMultiply(this.params.get(0)), new BiasAdd(this.params.get(1)),
				this.activation);
	}

	@Override
	public Layer deepCopy() {

		DenseLayer dl = new DenseLayer(this.neurons, this.activation.deepCopy());
		dl.first = this.first;
		dl.params = this.params == null ? null : this.params.stream().map(DoubleData::deepCopy).toList();
		dl.paramGrads = this.paramGrads == null ? null : this.paramGrads.stream().map(DoubleData::deepCopy).toList();
		dl.operations = this.operations == null ? null : this.operations.stream().map(Operation::deepCopy).toList();
		dl.seed = this.seed;

		dl.input = this.input.deepCopy();
		dl.output = this.output.deepCopy();
		
		return dl;
	}
}
