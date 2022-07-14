package in.dljava.operations;

import in.dljava.data.DoubleData;

public class Linear extends Operation {

	@Override
	public DoubleData output() {
	
		return this.input;
	}
	
	@Override
	public DoubleData inputGradient(DoubleData outGradient) {
		return outGradient;
	}
	
	@Override
	public Linear deepCopy() {
		
		Linear linear = new Linear();
		linear.input = linear.input.deepCopy();
		linear.out = linear.out.deepCopy();
		linear.inpGradient = linear.inpGradient.deepCopy();
	
		return linear;
	}
}
