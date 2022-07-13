package in.dljava.functions.optimizer;

public class MomentumOptimizer implements OptimizerFunction {

	private double learningRate;

	public MomentumOptimizer(double learningRate) {
		this.learningRate = learningRate;
	}

	public double optimize() {
		return this.learningRate * 0.9;
	}
}