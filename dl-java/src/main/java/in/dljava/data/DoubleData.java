package in.dljava.data;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class DoubleData implements Data, Cloneable {

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

	public DoubleData subDataNth(int n) {

		Shape newShape = this.shape.oneOf();
		int size = newShape.total();

		double[] sub = new double[size];
		System.arraycopy(this.data, n * size, sub, 0, size);
		return new DoubleData(newShape, sub);
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
			int[] dim = new int[dimensions.length - 1];
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

	public DoubleData deepCopy() {

		double[] cloned = new double[this.data.length];

		System.arraycopy(this.data, 0, cloned, 0, this.data.length);

		return new DoubleData(shape, cloned);
	}

	public DoubleData divide(double batchSize) {

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).div(batchSize).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = this.data[i] / batchSize;
		}

		return new DoubleData(this.shape, result);
	}

	public DoubleData subtract(DoubleData d) {

		if (!this.shape.equals(d.getShape()))
			throw new DataException("Cannot subtract data of shape " + this.shape + " to " + d.shape);

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).sub(DoubleVector.fromArray(SPECIES, d.data, i))
					.intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = this.data[i] - d.data[i];
		}

		return new DoubleData(this.shape, result);
	}

	public DoubleData transpose() {

		if (this.shape.dimensions().length != 2)
			throw new DataException("Cannot tranpose if it is not a matrix data : " + this.shape);

		var dim = this.shape.dimensions();

		var ndata = new double[this.data.length];
		var nShape = new Shape(dim[1], dim[0]);

		if (dim[0] == 1 || dim[1] == 1) {
			System.arraycopy(this.data, 0, ndata, 0, this.data.length);

		} else {

			for (int i = 0; i < dim[0]; i++) {
				for (int j = 0; j < dim[1]; j++) {
					ndata[i + (j * dim[0])] = data[j + (i * dim[1])];
				}
			}
		}

		return new DoubleData(nShape, ndata);
	}

	public DoubleData onesLike() {
		var ndata = new double[this.data.length];

		Arrays.fill(ndata, 1d);

		return new DoubleData(this.shape, ndata);
	}

	public DoubleData power(int num) {

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).pow(num).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = Math.pow(this.data[i], num);
		}

		return new DoubleData(this.shape, result);
	}

	public double average() {
		double total = 0;
		for (int i = 0; i < this.data.length; i++) {
			total += this.data[i];
		}

		return total / this.shape.total();
	}

	public DoubleData multiply(double d) {
		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).mul(d).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = this.data[i] * d;
		}

		return new DoubleData(this.shape, result);
	}

	public DoubleData inplaceSubtract(DoubleData d) {

		if (!this.shape.equals(d.getShape()))
			throw new DataException("Cannot subtract data of shape " + this.shape + " to " + d.shape);

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).sub(DoubleVector.fromArray(SPECIES, d.data, i))
					.intoArray(this.data, i);
		}

		for (; i < this.data.length; i++) {
			this.data[i] -= d.data[i];
		}

		return this;
	}

	public DoubleData multiply(DoubleData d) {

		if (!this.shape.equals(d.getShape()))
			throw new DataException("Cannot add data of shape " + this.shape + " to " + d.shape);

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).mul(DoubleVector.fromArray(SPECIES, d.data, i))
					.intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = this.data[i] * d.data[i];
		}

		return new DoubleData(this.shape, result);
	}

	public DoubleData exp() {
		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).lanewise(VectorOperators.EXP).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = Math.exp(this.data[i]);
		}

		return new DoubleData(this.shape, result);
	}

	public double total() {

		double total = 0;
		for (int i = 0; i < this.data.length; i++) {
			total += this.data[i];
		}

		return total;
	}

	public DoubleData diagFlat() {

		int s = this.shape.total();

		double[] d = new double[s * s];
		for (int i = 0; i < s; i++) {
			d[(i * s) + i] = this.data[i];
		}

		return new DoubleData(new Shape(s, s), d);
	}

	public DoubleData concatenate(DoubleData d, int axis) {

		int[] dim = this.shape.dimensions();
		int[] tdim = d.shape.dimensions();

		for (int i = axis + 1; i < dim.length; i++) {
			if (dim[i] != tdim[i]) {
				throw new DataException("Shape doesn't match with the data : " + shape + " with length : " + d.shape
						+ " on axis :" + axis);
			}
		}

		dim[axis] += tdim[axis];
		Shape s = new Shape(dim);
		int[] srcDim = this.shape.dimensions();

		double[] newData = new double[s.total()];
		if (axis == 0) {
			System.arraycopy(this.data, 0, newData, 0, this.data.length);
			System.arraycopy(d.data, 0, newData, this.data.length, d.data.length);
		} else if (axis == 1) {
			for (int i = 0; i < newData.length; i++) {
				boolean first = i % dim[1] < srcDim[1];
				
				newData[i] = (first ? this.data : d.data)[((i / dim[1]) * (first ? srcDim[1] : tdim[1])) + (i % dim[1])
						- (first ? 0 : srcDim[1])];
			}
		}

		return new DoubleData(s, newData);
	}
}