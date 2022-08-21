package in.dljava.operations;

import in.dljava.data.DoubleData;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Relu extends Operation {

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	@Override
	public DoubleData output(boolean inference) {

		var data = this.input.getData();
		var outdata = new double[data.length];
		for (int i = 0; i < data.length; i++) {
			outdata[i] = data[i] < 0d ? 0d : data[i];
		}

		return new DoubleData(this.input.getShape(), outdata);
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {
		int i = 0;
		var data = this.out.getData();
		var outdata = new double[data.length];

		for (; i < data.length; i++) {
			outdata[i] = (data[i] <= 0d ? 0d : 1d) * outGradient.getData()[i];
		}

		return new DoubleData(this.out.getShape(), outdata);
	}

	@Override
	public Relu deepCopy() {

		Relu relu = new Relu();
		relu.input = relu.input.deepCopy();
		relu.out = relu.out.deepCopy();
		relu.inpGradient = relu.inpGradient.deepCopy();

		return relu;
	}
}
