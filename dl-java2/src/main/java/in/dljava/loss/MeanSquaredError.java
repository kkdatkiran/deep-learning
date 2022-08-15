package in.dljava.loss;

import in.dljava.data.DoubleData;

public class MeanSquaredError extends Loss {

	private boolean normalized;

	public MeanSquaredError() {
		this(false);
	}

	public MeanSquaredError(boolean normalized) {
		this.normalized = normalized;
	}

	@Override
	public DoubleData inputGradient() {

		return this.prediction.subtract(this.target).divide(this.prediction.getShape().dimensions()[0]).multiply(2.0d);
	}

	@Override
	public double output() {

		DoubleData pred = this.prediction;

		if (this.normalized)
			pred = pred.divide(pred.sum(1, true));

		return pred.subtract(this.target).power(2).average();
	}

	@Override
	public MeanSquaredError deepCopy() {

		MeanSquaredError mse = new MeanSquaredError();

		mse.prediction = this.prediction.deepCopy();
		mse.target = this.target.deepCopy();
		mse.inpGradient = this.inpGradient.deepCopy();
		mse.normalized = this.normalized;

		return mse;
	}

}
