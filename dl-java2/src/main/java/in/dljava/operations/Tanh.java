package in.dljava.operations;

import in.dljava.data.DoubleData;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Tanh extends Operation {

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	@Override
	public DoubleData output(boolean inference) {

		var data = this.input.getData();
		var outdata = new double[data.length];

		int i = 0;
		int upperBound = SPECIES.loopBound(data.length);
		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, data, i).lanewise(VectorOperators.TANH).intoArray(outdata, i);
		}

		for (; i < data.length; i++) {
			outdata[i] = Math.tanh(data[i]);
		}

		return new DoubleData(this.input.getShape(), outdata);
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {

		var data = this.out.getData();
		var outdata = new double[data.length];

		int i = 0;
		int upperBound = SPECIES.loopBound(data.length);

		for (; i < upperBound; i += SPECIES.length()) {
			var vo = DoubleVector.fromArray(SPECIES, outGradient.getData(), i);
			DoubleVector.fromArray(SPECIES, data, i).lanewise(VectorOperators.POW, 2).neg().add(1).mul(vo)
					.intoArray(outdata, i);
		}

		for (; i < data.length; i++) {
			outdata[i] = (1 - (data[i] * data[i])) * outGradient.getData()[i];
		}

		return new DoubleData(this.out.getShape(), outdata);
	}

	@Override
	public Tanh deepCopy() {

		Tanh sigmoid = new Tanh();
		sigmoid.input = sigmoid.input.deepCopy();
		sigmoid.out = sigmoid.out.deepCopy();
		sigmoid.inpGradient = sigmoid.inpGradient.deepCopy();

		return sigmoid;
	}
}
