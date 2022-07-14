package in.dljava.optimizer;

import in.dljava.model.Sequential;

public abstract class Optimizer {

	protected double learningRate;
	protected Sequential model;
	
	public Optimizer(double learningRate) {
		this.learningRate = learningRate;
	}
	
	public void setModel(Sequential model) {
		this.model = model;
	}
	
	public abstract void step();
}
