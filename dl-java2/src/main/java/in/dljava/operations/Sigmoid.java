package in.dljava.operations;

import in.dljava.data.DoubleData;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Sigmoid extends Operation {

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	@Override
	public DoubleData output(boolean inference) {

		return this.input.onesLike().divide(this.input.neg().exp().add(1d));
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {

		return this.out.multiply(this.out.onesLike().subtract(this.out)).multiply(outGradient);
	}

	@Override
	public Sigmoid deepCopy() {

		Sigmoid sigmoid = new Sigmoid();
		sigmoid.input = sigmoid.input.deepCopy();
		sigmoid.out = sigmoid.out.deepCopy();
		sigmoid.inpGradient = sigmoid.inpGradient.deepCopy();

		return sigmoid;
	}
}
