package in.dljava.loss;

import in.dljava.data.DoubleData;

public class CrossEntropy extends Loss {

	private static final double EPSILON = 1e-9;

	private boolean singleClass;

	private DoubleData softmaxPreds;

	public CrossEntropy() {
		this.singleClass = false;
	}

	@Override
	protected double output() {
		if (this.target.getShape().dimensions()[1] == 1)
			this.singleClass = true;
		
		var localPred = this.prediction;

		if (singleClass) {
			localPred = localPred.concatenate(localPred.onesLike().subtract(localPred), 1);
			this.target = this.target.concatenate(this.target.onesLike().subtract(this.target), 1);
		}

		var softmaxPredsLocal = localPred.softmax(1);

		this.softmaxPreds = softmaxPredsLocal.clip(EPSILON, 1 - EPSILON);

		var softmaxCrossEntropyLoss = this.target.multiply(this.softmaxPreds.log()).neg()
				.subtract(this.target.onesLike().subtract(this.target)
						.multiply(this.softmaxPreds.onesLike().subtract(this.softmaxPreds).log()));
		return softmaxCrossEntropyLoss.sum(null, false).getData()[0] / localPred.getShape().dimensions()[0];

	}

	@Override
	protected DoubleData inputGradient() {

		if (this.singleClass)
			return this.softmaxPreds.subtract(this.target).unnormalize();
		
		return this.softmaxPreds.subtract(this.target).divide(this.prediction.getShape().dimensions()[0]);
	}

	@Override
	public CrossEntropy deepCopy() {

		CrossEntropy ce = new CrossEntropy();

		ce.prediction = this.prediction.deepCopy();
		ce.target = this.target.deepCopy();
		ce.inpGradient = this.inpGradient.deepCopy();
		ce.singleClass = this.singleClass;
		ce.softmaxPreds = this.softmaxPreds.deepCopy();

		return ce;
	}

}
