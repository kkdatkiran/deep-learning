package in.dljava.optimizer;

import in.dljava.model.Sequential;

public abstract class Optimizer {

	protected double learningRate;
	protected double finalLearningRate = 0;
	protected String decayType;
	protected boolean first = true;
	
	protected Sequential model;
	
	public Optimizer(double learningRate, Double finalLearningRate, String decayType) {
		this.learningRate = learningRate;
		if (finalLearningRate != null)
			this.finalLearningRate = finalLearningRate;
		this.decayType = decayType;
	}
	
	public void setModel(Sequential model) {
		this.model = model;
	}
	
	public abstract void step();
}
