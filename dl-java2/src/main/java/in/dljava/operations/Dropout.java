package in.dljava.operations;

import in.dljava.data.DoubleData;

public class Dropout extends Operation {

	private double prob;
	private DoubleData mask;

	public Dropout(double prob) {
		this.prob = prob;
	}

	@Override
	public DoubleData output(boolean inference) {

		if (inference)
			return this.input.multiply(this.prob);

		this.mask = DoubleData.binomial(1, this.prob, this.input.getShape().deepCopy());

		return this.input.multiply(this.mask);
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {

		if (this.mask == null)
			return outGradient;

		return outGradient.multiply(this.mask);
	}

	@Override
	public Dropout deepCopy() {
		Dropout dropout = new Dropout(this.prob);
		dropout.inpGradient = this.inpGradient.deepCopy();
		dropout.input = this.input.deepCopy();
		dropout.out = this.out.deepCopy();
		return dropout;
	}

}
