package in.dljava.loss;

import in.dljava.DLException;
import in.dljava.data.DoubleData;

public abstract class Loss {
	
	protected DoubleData prediction;
	protected DoubleData target;
	protected DoubleData inpGradient;
	
	public double forward(DoubleData prediction, DoubleData target) {
		
		if (!prediction.getShape().equals(target.getShape()))
			throw new DLException("Mismatch sizes "+prediction.getShape()+" with "+target.getShape());
		
		this.prediction = prediction;
		this.target = target;
		
		return this.output();
	}
	
	public DoubleData backward() {
		this.inpGradient = this.inputGradient();
		
		if (!prediction.getShape().equals(inpGradient.getShape()))
			throw new DLException("Mismatch sizes "+prediction.getShape()+" with "+inpGradient.getShape());
		
		return this.inpGradient;
		
	}

	protected abstract double output();
	protected abstract DoubleData inputGradient();
	public abstract Loss deepCopy();
}
