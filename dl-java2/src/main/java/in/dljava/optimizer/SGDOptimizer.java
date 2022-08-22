package in.dljava.optimizer;

import java.util.Map;

import in.dljava.data.DoubleData;

public class SGDOptimizer extends Optimizer {

	public SGDOptimizer(double learningRate) {
		super(learningRate, null, null);
	}

	@Override
	public void updatRule(Map<String, DoubleData> arg) {
	
		DoubleData param = arg.get(PARAM);
		DoubleData paramGrad = arg.get(PARAM_GRAD);
		
		param.inplaceSubtract(paramGrad.multiply(this.learningRate));
	}
}
