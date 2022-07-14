package in.dljava.operations;

import in.dljava.data.DoubleData;

public class WeightMultiply extends ParameterOperation{

	public WeightMultiply(DoubleData w) {
		super(w);
	}
	
	@Override
	public DoubleData output() {
		return this.input.matrixMultiply(this.param);
	}
	
	@Override
	public DoubleData inputGradient(DoubleData outGradient) {
		
		return outGradient.matrixMultiply(this.param.transpose());
	}
	
	@Override
	public DoubleData parameterGradient(DoubleData outGradient) {
		return this.input.transpose().matrixMultiply(outGradient);
	}
	
	@Override
	public WeightMultiply deepCopy() {
		
		WeightMultiply w = new WeightMultiply(this.param.deepCopy());
		w.input = this.input.deepCopy();
		w.out = this.out.deepCopy();
		w.inpGradient = this.inpGradient.deepCopy();
		w.paramGrad = this.paramGrad.deepCopy();
		
		return w;
	}
}
