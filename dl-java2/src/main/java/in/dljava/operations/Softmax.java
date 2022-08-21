package in.dljava.operations;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Softmax extends Operation {

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	@Override
	public DoubleData output(boolean inference) {

		var data = this.input.getData();
		var sum = this.input.exp().total();

		var outdata = new double[data.length];
		for (int i = 0; i < data.length; i++) {
			outdata[i] = Math.exp(data[i]) / sum;
		}

		return new DoubleData(this.input.getShape(), outdata);
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {

		var s = this.out.deepCopy().reShape(new Shape(this.out.getShape().total(), 1));
		var jac = s.diagFlat().subtract(s.matrixMultiply(s.transpose()));
		return outGradient.matrixMultiply(jac);
	}

	@Override
	public Softmax deepCopy() {

		Softmax softmax = new Softmax();
		softmax.input = softmax.input.deepCopy();
		softmax.out = softmax.out.deepCopy();
		softmax.inpGradient = softmax.inpGradient.deepCopy();

		return softmax;
	}
}
