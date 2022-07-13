package in.dljava.activation;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

public class Sigmoid implements ActivationFunction {

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	@Override
	public void apply(double[] data) {

		int i;

		for (i = 0; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
			var vb = DoubleVector.fromArray(SPECIES, data, i);
			vb.neg().lanewise(VectorOperators.EXP).add(1d).pow(-1).intoArray(data, i);
		}

		for (; i < data.length; i++) {
			data[i] = 1d / (1d + Math.exp(-data[i]));
		}
	}

	@Override
	public void applyDerivative(double[] data) {
		int i;

		for (i = 0; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
			var vb = DoubleVector.fromArray(SPECIES, data, i);
			vb.mul(vb.neg().add(1d)).intoArray(data, i);
		}

		for (; i < data.length; i++) {
			data[i] = data[i] * (1 - data[i]);
		}
	}

}
