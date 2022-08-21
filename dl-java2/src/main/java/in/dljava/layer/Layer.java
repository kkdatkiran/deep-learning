package in.dljava.layer;

import java.util.ArrayList;
import java.util.List;

import in.dljava.DLException;
import in.dljava.data.DoubleData;
import in.dljava.operations.Operation;
import in.dljava.operations.ParameterOperation;

public abstract class Layer {

	protected int neurons;
	protected boolean first;
	protected List<DoubleData> params;
	protected List<DoubleData> paramGrads;
	protected List<Operation> operations;
	protected DoubleData input;
	protected DoubleData output;
	protected int seed = 0;

	protected Layer(int neurons) {

		this.neurons = neurons;
		this.first = true;
		this.params = new ArrayList<>();
		this.paramGrads = new ArrayList<>();
		this.operations = new ArrayList<>();
	}

	public abstract void setupLayer(DoubleData numIn);

	public DoubleData forward(DoubleData input, boolean inference) {

		if (this.first) {
			this.setupLayer(input);
			this.first = false;
		}

		this.input = input;

		for (Operation operation : this.operations) {
			input = operation.forward(input, inference);
		}

		this.output = input;

		return this.output;
	}

	public DoubleData backward(DoubleData outputGradient) {

		if (!this.output.getShape().equals(outputGradient.getShape()))
			throw new DLException("Mismatch sizes "+this.output.getShape()+" with "+outputGradient.getShape());
		
		for (int i = this.operations.size() - 1; i >= 0; i--) {
			outputGradient = operations.get(i).backward(outputGradient);
		}
		this.paramGradientAccumulate();

		return outputGradient;
	}

	public void paramGradientAccumulate() {
		this.paramGrads = new ArrayList<>();
		for (Operation operation : this.operations) {
			if (operation instanceof ParameterOperation op) {
				this.paramGrads.add(op.getParamGrad());
			}
		}
	}

	public void paramsAccumulate() {

		this.params = new ArrayList<>();
		for (Operation operation : this.operations) {
			if (operation instanceof ParameterOperation op) {
				this.paramGrads.add(op.getParam());
			}
		}
	}

	public void setSeed(int seed) {
		this.seed = seed;
	}

	public List<DoubleData> getParams() {
		return params;
	}
	
	public List<DoubleData> getParamGrads() {
		return paramGrads;
	}

	public void setFirst(boolean b) {
		this.first = b;
		
	}
	
	public abstract Layer deepCopy();
}
