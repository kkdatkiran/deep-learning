package in.dljava.optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import in.dljava.data.DoubleData;
import in.dljava.util.ZipUtil;

public class SGDMomentum extends Optimizer {

	private static final String VELOCITY = "velocity";
	
	private double momentum;
	private List<DoubleData> velocities;

	public SGDMomentum(double learningRate, Double finalLearningRate, String decayType, double momentum) {

		super(learningRate, finalLearningRate, decayType);
		this.momentum = momentum;
	}

	@Override
	public void step(int epoch) {

		if (this.first) {

			this.velocities = new ArrayList<>();
			this.model.params().forEachRemaining(e -> this.velocities.add(e.zerosLike()));
			this.first = false;
		}

		ZipUtil.zip(this.model.params(), this.model.paramGrads(), this.velocities.iterator()).stream().forEach(
				tup -> this.updatRule(Map.of(PARAM, tup.getT1(), PARAM_GRAD, tup.getT2(), VELOCITY, tup.getT3())));
	}

	@Override
	public void updatRule(Map<String, DoubleData> arg) {

		var velocity = arg.get(VELOCITY);
		var param = arg.get(PARAM);
		var paramGrad = arg.get(PARAM_GRAD);

		velocity.inplaceMultiply(this.momentum);
		velocity.inplaceAdd(paramGrad.multiply(this.learningRate));
		param.inplaceSubtract(velocity);
	}
}
