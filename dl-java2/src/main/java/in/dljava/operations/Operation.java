package in.dljava.operations;

import in.dljava.DLException;
import in.dljava.data.DoubleData;

public abstract class Operation {
	
	protected DoubleData input;
	protected DoubleData out;
	protected DoubleData inpGradient;

	public DoubleData forward(DoubleData input) {
		this.input = input;
		this.out = this.output();
		return this.out;
	}
	
	public DoubleData backward(DoubleData outGradient)  {
		if (!this.out.getShape().equals(outGradient.getShape()))
			throw new DLException("Mismatch sizes "+this.out.getShape()+" with "+outGradient.getShape());
		
		this.inpGradient = this.inputGradient(outGradient);
		
		if (!this.input.getShape().equals(inpGradient.getShape()))
			throw new DLException("Mismatch sizes "+this.input.getShape()+" with "+inpGradient.getShape());
		return this.inpGradient;
	}
	
	
	public abstract DoubleData output();
	public abstract DoubleData inputGradient(DoubleData outGradient);
	public abstract Operation deepCopy();

}
