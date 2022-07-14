package in.dljava.loss;

import in.dljava.data.DoubleData;

public class MeanSquaredError extends Loss {

	@Override
	public DoubleData inputGradient() {

		return this.prediction.subtract(this.target).divide(this.prediction.getShape().dimensions()[0]).multiply(2.0d);
	}

	@Override
	public double output() {

		return this.prediction.subtract(this.target).power(2).average();
	}

	@Override
	public MeanSquaredError deepCopy() {
		
		MeanSquaredError mse = new MeanSquaredError();
		
		mse.prediction = this.prediction.deepCopy();
		mse.target = this.target.deepCopy();
		mse.inpGradient = this.inpGradient.deepCopy();
		
		return mse;
	}

}
