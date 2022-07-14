package in.dljava.operations;

import in.dljava.DLException;
import in.dljava.data.DoubleData;

public abstract class ParameterOperation extends Operation{
	
	
	protected DoubleData param;
	protected DoubleData paramGrad;
	
	protected ParameterOperation(DoubleData param) {
		
		this.param = param;
	}
	
	@Override
	public DoubleData backward(DoubleData outGradient) {
		
		if (!this.out.getShape().equals(outGradient.getShape()))
			throw new DLException("Mismatch sizes "+this.out.getShape()+" with "+outGradient.getShape());
		
		this.inpGradient = this.inputGradient(outGradient);
		this.paramGrad = this.parameterGradient(outGradient);
		
		if (!this.input.getShape().equals(inpGradient.getShape()))
			throw new DLException("Mismatch sizes "+this.input.getShape()+" with "+this.inpGradient.getShape());
		
		if (!this.param.getShape().equals(paramGrad.getShape()))
			throw new DLException("Mismatch sizes "+this.param.getShape()+" with "+this.paramGrad.getShape());
		
		return this.inpGradient;
	}

	public abstract DoubleData parameterGradient(DoubleData outGradient);

	public DoubleData getParamGrad() {
		return this.paramGrad;
	}

	public DoubleData getParam() {
		return this.param;
	}
}
