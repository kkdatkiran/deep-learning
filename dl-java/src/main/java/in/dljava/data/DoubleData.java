package in.dljava.data;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class DoubleData implements Data {

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	private double[] data;
	private Shape shape;

	public DoubleData(Shape shape, double[] data) {

		if (shape.total() != data.length)
			throw new DataException("Shape doesn't match with the data : " + shape + " with length : " + data.length);

		this.shape = shape;
		this.data = data;
	}

	public DoubleData(Shape s) {
		this(s, null);
	}

	public double[] getData() {

		return this.data;
	}

	public DoubleData setData(double[] data) {

		if (shape.total() != data.length)
			throw new DataException("Shape doesn't match with the data");

		this.data = data;
		return this;
	}

	public DoubleData reShape(Shape shape) {
		this.shape = shape;
		return this;
	}

	@Override
	public Shape getShape() {
		return this.shape;
	}

	@Override
	public String toString() {

		return shape + "\nData : " + Arrays.toString(data);
	}

	public DoubleData matrixMultiply(DoubleData d) {

		if (this.shape.dimensions().length != 2 || d.shape.dimensions().length != 2)
			throw new DataException("Only 2 dimension data can be matrix multiplied");

		if (this.shape.dimensions()[1] != d.shape.dimensions()[0])
			throw new DataException(
					"Matrix multiplication cannot be done for the data " + this.shape + " and " + d.shape);

		double[] result = new double[this.shape.dimensions()[0] * d.shape.dimensions()[1]];

		for (int i = 0; i < this.shape.dimensions()[0]; i++) {

			int indexCbase = i * d.shape.dimensions()[1];

			double valA = this.data[i * this.shape.dimensions()[1]];
			int j;
			for (j = 0; j < SPECIES.loopBound(d.shape.dimensions()[1]); j += SPECIES.length()) {
				var vb = DoubleVector.fromArray(SPECIES, d.data, j);
				vb.mul(valA).intoArray(result, indexCbase + j);
			}
			for (; j < d.shape.dimensions()[1]; j++) {
				result[indexCbase + j] = valA * d.data[j];
			}

			for (int k = 1; k < d.shape.dimensions()[0]; k++) {
				int indexB = k * d.shape.dimensions()[1];

				valA = this.data[i * this.shape.dimensions()[1] + k];

				for (j = 0; j < SPECIES.loopBound(d.shape.dimensions()[1]); j += SPECIES.length()) {
					var vb = DoubleVector.fromArray(SPECIES, d.data, indexB + j);
					var vc = DoubleVector.fromArray(SPECIES, result, indexCbase + j);
					vc.add(vb.mul(valA)).intoArray(result, indexCbase + j);
				}

				for (; j < d.shape.dimensions()[1]; j++) {
					result[indexCbase + j] += valA * d.data[indexB + j];
				}
			}
		}

		return new DoubleData(new Shape(this.shape.dimensions()[0], d.shape.dimensions()[1]), result);
	}

	public DoubleData add(DoubleData d) {

		if (!this.shape.equals(d.getShape()))
			throw new DataException("Cannot add data of shape " + this.shape + " to " + d.shape);

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).add(DoubleVector.fromArray(SPECIES, d.data, i))
					.intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = this.data[i] + d.data[i];
		}

		return new DoubleData(this.shape, result);
	}

	@Override
	public Data subData(Shape shape, int from, int to) {

		double[] sub = new double[to - from];
		System.arraycopy(this.data, from, sub, 0, to - from);
		return new DoubleData(shape, sub);
	}

	@Override
	public void print() {

		this.print(this.shape.dimensions(), 0, this.data.length);
	}

	private void print(int[] dimensions, int from, int to) {

		StringBuilder sb = new StringBuilder();

		if (dimensions.length == 1) {

			sb.append("[");
			for (int i = from; i < to; i++) {
				sb.append(String.format("%11.8f%n ", this.data[i]));
			}
			sb.append("]");
			System.out.println(sb);
		} else if (dimensions.length == 2) {

			for (int i = 0; i < dimensions[0]; i++) {
				sb.append("|");
				for (int j = 0; j < dimensions[1]; j++) {
					sb.append(String.format("%11.8f ", this.data[from++]));
				}
				sb.append("|\n");
			}
			System.out.println(sb);
		} else {

			int matrixSize = dimensions[0];
			int dim[] = new int[dimensions.length - 1];
			dim[0] = dimensions[0];

			for (int i = 1; i < dimensions.length - 1; i++) {
				matrixSize *= dimensions[i];
				dim[i] = dimensions[i];
			}

			for (int i = 0; i < dimensions[0]; i++) {
				System.out.println(0 + Stream.of(dim).map(Object::toString).collect(Collectors.joining("x", "x", "")));
				this.print(dim, from + (i * matrixSize), from + ((i + 1) * matrixSize));
			}
		}

	}
}