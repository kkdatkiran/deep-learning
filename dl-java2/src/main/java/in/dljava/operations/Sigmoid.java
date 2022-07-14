package in.dljava.operations;

import in.dljava.data.DoubleData;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Sigmoid extends Operation {
	
	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	
	@Override
	public DoubleData output() {
		
		
		int i;
		
		var data = this.input.getData();
		var outdata = new double[data.length];

		for (i = 0; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
			var vb = DoubleVector.fromArray(SPECIES, data, i);
			vb.neg().lanewise(VectorOperators.EXP).add(1d).pow(-1).intoArray(outdata, i);
		}

		for (; i < data.length; i++) {
			outdata[i] = 1d / (1d + Math.exp(-data[i]));
		}
		
		return new DoubleData(this.input.getShape(), outdata);
	}
	
	@Override
	public DoubleData inputGradient(DoubleData outGradient) {
		int i;
		var data = this.out.getData();
		var outdata = new double[data.length];

		for (i = 0; i < SPECIES.loopBound(data.length); i += SPECIES.length()) {
			var vb = DoubleVector.fromArray(SPECIES, data, i);
			var vo = DoubleVector.fromArray(SPECIES, outGradient.getData(), i);
			vb.mul(vb.neg().add(1d)).mul(vo).intoArray(outdata, i);
		}

		for (; i < data.length; i++) {
			outdata[i] = data[i] * (1 - data[i]) *outGradient.getData()[i];
		}
		
		return new DoubleData(this.out.getShape(), outdata);
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
