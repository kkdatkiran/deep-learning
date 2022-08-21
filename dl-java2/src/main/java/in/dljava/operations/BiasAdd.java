package in.dljava.operations;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;

public class BiasAdd extends ParameterOperation {

	public BiasAdd(DoubleData params) {

		super(params);
	}

	@Override
	public DoubleData output(boolean inference) {
		return this.input.add(this.param);
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {
		return this.input.onesLike().multiply(outGradient);
	}

	@Override
	public DoubleData parameterGradient(DoubleData outGradient) {
		this.paramGrad = this.param.onesLike().multiply(outGradient);
		return this.paramGrad.reShape(new Shape(1, this.paramGrad.getShape().dimensions()[1]));
	}
	
	@Override
	public BiasAdd deepCopy() {
		
		BiasAdd b = new BiasAdd(this.param.deepCopy());
		b.input = this.input.deepCopy();
		b.out = this.out.deepCopy();
		b.inpGradient = this.inpGradient.deepCopy();
		b.paramGrad = this.paramGrad.deepCopy();
		
		return b;
	}
}
