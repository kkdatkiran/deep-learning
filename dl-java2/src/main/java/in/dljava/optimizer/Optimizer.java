package in.dljava.optimizer;

import java.util.Map;

import in.dljava.data.DoubleData;
import in.dljava.model.Sequential;
import in.dljava.util.ZipUtil;

public abstract class Optimizer {
	
	public static final String PARAM = "param";
	public static final String PARAM_GRAD = "paramGrad";

	protected double learningRate;
	protected double finalLearningRate = 0;
	protected String decayType;
	protected boolean first = true;
	private double decayPerEpoch;
	private int maxEpochs;

	protected Sequential model;

	protected Optimizer(double learningRate, Double finalLearningRate, String decayType) {
		this.learningRate = learningRate;
		if (finalLearningRate != null)
			this.finalLearningRate = finalLearningRate;
		this.decayType = decayType;
	}

	public void setModel(Sequential model) {
		this.model = model;
	}

	public void setupDecay() {

		if (this.decayType == null)
			return;

		if (this.decayType.equalsIgnoreCase("linear"))
			this.decayPerEpoch = ((this.learningRate - this.finalLearningRate) / (this.maxEpochs - 1));

		this.decayPerEpoch = Math.pow(this.finalLearningRate / this.learningRate, 1d / (this.maxEpochs - 1));
	}

	public void decayLearningRate() {

		if (this.decayType == null)
			return;

		if (this.decayType.equalsIgnoreCase("linear"))
			this.learningRate -= this.decayPerEpoch;

		this.learningRate *= this.decayPerEpoch;
	}

	public void step(int epoch) {

		ZipUtil.zip(this.model.params(), this.model.paramGrads())
				.forEach(e -> this.updatRule(Map.of(PARAM, e.getT1(), PARAM_GRAD, e.getT2())));
	}

	public abstract void updatRule(Map<String, DoubleData> arg);

	public double getDecayPerEpoch() {
		return decayPerEpoch;
	}

	public void setDecayPerEpoch(double decayPerEpoch) {
		this.decayPerEpoch = decayPerEpoch;
	}

	public int getMaxEpochs() {
		return maxEpochs;
	}

	public void setMaxEpochs(int maxEpochs) {
		this.maxEpochs = maxEpochs;
	}

	public double getFinalLearningRate() {
		return this.finalLearningRate;
	}
}
