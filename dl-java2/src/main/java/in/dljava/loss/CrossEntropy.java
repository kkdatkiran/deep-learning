package in.dljava.loss;

import in.dljava.data.DoubleData;

public class CrossEntropy extends Loss{
	
	private static final double EPSILON = 1e-9;
	
	private boolean singleClass;
	
	public CrossEntropy() {
		this.singleClass = false;
	}

	@Override
	protected double output() {
		if (this.target.getShape().dimensions()[1] == 1)
			this.singleClass = true;
		
		if (singleClass) {
			this.prediction = this.prediction.concatenate(this.prediction.onesLike().subtract(this.prediction), 1);
			this.target = this.prediction.concatenate(this.target.onesLike().subtract(this.target), 1);
		}
		
//		var 
		return 1d;
	}

	@Override
	protected DoubleData inputGradient() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Loss deepCopy() {
		// TODO Auto-generated method stub
		return null;
	}

}
