package in.dljava.optimizer;

public class SGDOptimizer extends Optimizer {

	public SGDOptimizer(double learningRate) {
		super(learningRate);
	}

	@Override
	public void step() {

		this.model.zipParamsParamGrads().forEach(e -> e.getT1().inplaceSubtract(e.getT2()).multiply(this.learningRate));

	}
}
