package in.dljava.data;

import java.util.Arrays;
import java.util.stream.Collectors;

import in.dljava.util.ArraysUtil;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import lombok.NonNull;

public class DoubleData implements Data {

	static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

	private double[] data;
	private Shape shape;

	public DoubleData(@NonNull Shape shape, double[] data) {

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

	public DoubleData reShape(int... dims) {
		return this.reShape(new Shape(dims));
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
			return this.matrixMultiplyND(d);

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

	public DoubleData matrixMultiplyND(DoubleData d) {

		DoubleData a = this;
		DoubleData b = d;

		int supressDim = -1;

		if (a.shape.numberOfAxes() == 1 || b.shape.numberOfAxes() == 1) {
			DoubleData oned;
			DoubleData multid;

			if (a.shape.numberOfAxes() == 1) {
				a = a.deepCopy();
				oned = a;
				multid = b;
			} else {
				b = b.deepCopy();
				oned = b;
				multid = a;
			}

			int[] mdims = multid.shape.dimensions();
			int[] odims = oned.shape.dimensions();

			int[] tdims = new int[2];

			if (mdims[mdims.length - 1] == odims[0]) {
				tdims[0] = odims[0];
				tdims[1] = 1;
				supressDim = 0;
			} else if (mdims[mdims.length - 2] == odims[1]) {
				tdims[0] = 1;
				tdims[1] = odims[0];
				supressDim = 1;
			} else {
				throw new DataException("Cannot multiply the data with shape " + this.shape + " and " + d.shape);
			}

			oned.reShape(tdims);
		}

		if (a.shape.numberOfAxes() != b.shape.numberOfAxes()) {
			boolean asmall = a.shape.numberOfAxes() < b.shape.numberOfAxes();
			DoubleData small;
			DoubleData big;

			if (asmall) {
				a = (a == this) ? a.deepCopy() : a;
				small = a;
				big = b;
			} else {
				b = (b == d) ? b.deepCopy() : b;
				small = b;
				big = a;
			}

			int[] newDims = new int[big.shape.numberOfAxes()];
			int[] smallDims = small.shape.dimensions();
			Arrays.fill(newDims, 1);
			System.arraycopy(smallDims, 0, newDims, newDims.length - smallDims.length, smallDims.length);

			for (int i = newDims.length - 1, j = smallDims.length - 1; j >= 0; i--, j--) {
				newDims[i] = smallDims[j];
			}

			small.reShape(newDims);
		}

		if (a.shape.numberOfAxes() == 2)
			return a.matrixMultiply(b);

		var x = stretchToSameSize(a, b, 2);
		a = x[0];
		b = x[1];

		int[] finDims = a.shape.dimensions();
		int[] bDims = b.shape.dimensions();
		finDims[finDims.length - 1] = bDims[finDims.length - 1];

		Shape finShape = new Shape(finDims);

		double[] c = new double[finShape.total()];

		int[] aDims = a.shape.dimensions();

		int aRows = aDims[aDims.length - 2];
		int aColumns = aDims[aDims.length - 1];
		int aSkipper = aRows * aColumns;

		int bRows = bDims[bDims.length - 2];
		int bColumns = bDims[bDims.length - 1];
		int bSkipper = bRows * bColumns;

		if (aColumns != bRows) {
			throw new DataException("Cannot multiply the data with shape " + this.shape + " and " + d.shape);
		}

		int cRows = finDims[finDims.length - 2];
		int cColumns = finDims[finDims.length - 1];
		int cSkipper = cRows * cColumns;

		for (int aOffset = 0, bOffset = 0,
				cOffset = 0; aOffset < a.shape.total(); aOffset += aSkipper, bOffset += bSkipper, cOffset += cSkipper) {
			for (int i = 0; i < aRows; i++) {

				int indexCbase = i * bColumns;

				double valA = a.data[aOffset + i * aColumns];
				int j;
				for (j = 0; j < SPECIES.loopBound(bColumns); j += SPECIES.length()) {
					var vb = DoubleVector.fromArray(SPECIES, b.data, bOffset + j);
					vb.mul(valA).intoArray(c, cOffset + indexCbase + j);
				}
				for (; j < bColumns; j++) {
					c[cOffset + indexCbase + j] = valA * b.data[j + bOffset];
				}

				for (int k = 1; k < bRows; k++) {
					int indexB = k * bColumns;

					valA = a.data[i * aColumns + k + aOffset];

					for (j = 0; j < SPECIES.loopBound(bColumns); j += SPECIES.length()) {
						var vb = DoubleVector.fromArray(SPECIES, b.data, bOffset + indexB + j);
						var vc = DoubleVector.fromArray(SPECIES, c, cOffset + indexCbase + j);
						vc.add(vb.mul(valA)).intoArray(c, cOffset + indexCbase + j);
					}

					for (; j < bColumns; j++) {
						c[cOffset + indexCbase + j] += valA * b.data[bOffset + indexB + j];
					}
				}
			}
		}

		DoubleData result = new DoubleData(finShape, c);

		if (supressDim != -1)
			result.reShape(
					ArraysUtil.splice(result.shape.dimensions(), result.shape.numberOfAxes() - supressDim - 1, 1));

		return result;
	}

	public DoubleData add(DoubleData d) {

		if (this.data.length == 1) {

			return d.add(this.data[0]);
		} else if (d.data.length == 1) {

			return this.add(d.data[0]);
		}

		var x = stretchToSameSize(this, d);
		var a = x[0];
		var b = x[1];

		int upperBound = SPECIES.loopBound(a.data.length);

		int i = 0;
		double[] result = new double[a.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, a.data, i).add(DoubleVector.fromArray(SPECIES, b.data, i)).intoArray(result,
					i);
		}

		for (; i < this.data.length; i++) {
			result[i] = a.data[i] + b.data[i];
		}

		return new DoubleData(a.shape, result);
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

	public DoubleData subDataNthLooseFirstDim(int n) {

		Shape newShape = this.shape.oneOfLooseFirstDim();
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

			int matrixSize = 1;
			int dim[] = new int[dimensions.length - 1];
			dim[0] = dimensions[0];

			for (int i = 1; i < dimensions.length; i++) {
				matrixSize *= dimensions[i];
				dim[i - 1] = dimensions[i];
			}

			for (int i = 0; i < dimensions[0]; i++) {

				System.out.println(
						(i + 1) + Arrays.stream(dim).mapToObj(e -> "" + e).collect(Collectors.joining("x", "x", "")));
				this.print(dim, from + (i * matrixSize), from + ((i + 1) * matrixSize));
			}
		}

	}

	public DoubleData deepCopy() {

		double[] cloned = new double[this.data.length];

		System.arraycopy(this.data, 0, cloned, 0, this.data.length);

		return new DoubleData(shape, cloned);
	}

	public DoubleData divide(double num) {

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).div(num).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = this.data[i] / num;
		}

		return new DoubleData(this.shape, result);
	}

	public DoubleData subtract(DoubleData d) {

		if (!this.shape.equals(d.getShape()))
			throw new DataException("Cannot subtract data of shape " + d.shape + " from " + this.shape);

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

	public DoubleData transpose(int... axes) {

		int[] toDims = new int[axes.length];
		int[] fromDims = this.shape.dimensions();

		for (int i = 0; i < axes.length; i++) {
			toDims[i] = fromDims[axes[i]];
		}

		Shape toShape = new Shape(toDims);

		ArraysUtil.inplaceReverse(fromDims);
		NumberGenerator numgen = new NumberGenerator(fromDims, axes);

		double[] reshaped = new double[this.shape.total()];

		for (int i = 0; i < this.shape.total(); i++)
			reshaped[i] = this.data[numgen.nextNumber()];

		return new DoubleData(toShape, reshaped);
	}

	public DoubleData onesLike() {
		var ndata = new double[this.data.length];

		Arrays.fill(ndata, 1d);

		return new DoubleData(this.shape.deepCopy(), ndata);
	}

	public DoubleData zerosLike() {
		var ndata = new double[this.data.length];

		Arrays.fill(ndata, 0d);

		return new DoubleData(this.shape.deepCopy(), ndata);
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

	public DoubleData add(double d) {
		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).add(d).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = this.data[i] + d;
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

		if (this.data.length == 1) {

			return d.multiply(this.data[0]);
		} else if (d.data.length == 1) {

			return this.multiply(d.data[0]);
		}

		var x = stretchToSameSize(this, d);
		var a = x[0];
		var b = x[1];

		int upperBound = SPECIES.loopBound(a.data.length);

		int i = 0;
		double[] result = new double[a.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, a.data, i).mul(DoubleVector.fromArray(SPECIES, b.data, i)).intoArray(result,
					i);
		}

		for (; i < this.data.length; i++) {
			result[i] = a.data[i] * b.data[i];
		}

		return new DoubleData(a.shape, result);

	}

	public DoubleData divide(DoubleData d) {

		if (d.data.length == 1) {

			return this.divide(d.data[0]);
		}

		var x = stretchToSameSize(this, d);
		var a = x[0];
		var b = x[1];

		int upperBound = SPECIES.loopBound(a.data.length);

		int i = 0;
		double[] result = new double[a.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, a.data, i).div(DoubleVector.fromArray(SPECIES, b.data, i)).intoArray(result,
					i);
		}

		for (; i < this.data.length; i++) {
			result[i] = a.data[i] / b.data[i];
		}

		return new DoubleData(a.shape, result);

	}

	public static DoubleData[] stretchToSameSize(DoubleData a, DoubleData b) {
		return stretchToSameSize(a, b, 0);
	}

	public static DoubleData[] stretchToSameSize(DoubleData a, DoubleData b, int ignoreLastNDims) {

		if (!a.shape.equals(b.shape)) {

			var ashape = a.shape.dimensions();
			var bshape = b.shape.dimensions();
			a = a.deepCopy();
			b = b.deepCopy();

			if (ashape.length < bshape.length) {
				var x = new int[bshape.length];
				Arrays.fill(x, 1);
				System.arraycopy(ashape, 0, x, bshape.length - ashape.length, ashape.length);
				a.reShape(new Shape(x));
			} else if (bshape.length < ashape.length) {
				var x = new int[ashape.length];
				Arrays.fill(x, 1);
				System.arraycopy(bshape, 0, x, ashape.length - bshape.length, bshape.length);
				b.reShape(new Shape(x));
			}

			ashape = a.shape.dimensions();
			bshape = b.shape.dimensions();

			for (int i = 0; i < ashape.length - ignoreLastNDims; i++) {
				if (ashape[i] == bshape[i])
					continue;

				if (ashape[i] == 1) {
					a = a.stretchDimension(bshape.length - i - 1, bshape[i]);
				} else if (bshape[i] == 1) {
					b = b.stretchDimension(ashape.length - i - 1, ashape[i]);
				}

				ashape = a.shape.dimensions();
				bshape = b.shape.dimensions();
			}
		}

		return new DoubleData[] { a, b };
	}

	public static DoubleData full(Shape shape, double num) {

		double[] x = new double[shape.total()];

		Arrays.fill(x, num);

		return new DoubleData(shape, x);
	}

	public DoubleData stretchDimension(int axis, int times) {

		int[] dims = this.shape.dimensions();
		if (dims.length > 4)
			throw new DataException("Data with more than 4 dimensions cannot be stretched.");

		int[] newDims = this.shape.dimensions();
		newDims[dims.length - axis - 1] = times;

		for (int i = 0; i < dims.length / 2; i++) {
			int t = dims[i];
			dims[i] = dims[dims.length - i - 1];
			dims[dims.length - i - 1] = t;
		}

		int fourthDim = 1;
		if (dims.length == 4) {
			fourthDim = (axis == 3 ? times : dims[3]);
		}

		int thirdDim = 1;
		int eachFourthDimSize = 1;
		if (dims.length >= 3) {
			thirdDim = (axis == 2 ? times : dims[2]);
			eachFourthDimSize = dims[2];
		}

		int secondDim = 1;
		int eachThirdDimSize = 1;
		if (dims.length >= 2) {
			secondDim = (axis == 1 ? times : dims[1]);
			eachThirdDimSize = dims[1];
		}
		eachFourthDimSize *= eachThirdDimSize;

		int firstDim = 1;
		int eachSecondDimSize = 1;
		if (dims.length >= 1) {
			firstDim = (axis == 0 ? times : dims[0]);
			eachSecondDimSize = dims[0];
		}
		eachFourthDimSize *= eachSecondDimSize;
		eachThirdDimSize *= eachSecondDimSize;

		Shape newShape = new Shape(newDims);
		double[] newData = new double[newShape.total()];
		int ind = 0;

		for (int i = 0; i < fourthDim; i++) {

			int ii = (axis == 3 ? 0 : i);
			for (int j = 0; j < thirdDim; j++) {

				int jj = (axis == 2 ? 0 : j);
				for (int k = 0; k < secondDim; k++) {

					int kk = (axis == 1 ? 0 : k);
					for (int l = 0; l < firstDim; l++) {

						int ll = (axis == 0 ? 0 : l);
						int dind = (ii * eachFourthDimSize) + (jj * eachThirdDimSize) + (kk * eachSecondDimSize) + ll;
						newData[ind++] = this.data[dind];
					}
				}
			}
		}

		return new DoubleData(newShape, newData);
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

	public DoubleData concatenate(DoubleData d, Integer axis) {

		if (axis == null) {
			int total = this.shape.total() + d.shape.total();
			double[] darray = new double[total];
			System.arraycopy(this.data, 0, darray, 0, this.data.length);
			System.arraycopy(d.data, 0, darray, this.data.length, d.data.length);
			return new DoubleData(new Shape(total), darray);
		}

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

	public DoubleData sum(Integer axis, boolean keepDimensions) {

		var dims = this.shape.dimensions();

		if (axis != null && axis >= dims.length) {
			throw new DataException("Cannot perform sum on the axis : " + axis);
		}

		double[] result;

		Shape resultShape = null;

		if (axis == null) {
			result = new double[1];
			for (int i = 0; i < this.data.length; i++) {
				result[0] += this.data[i];
			}
			if (!keepDimensions) {
				resultShape = new Shape(1);
			} else {
				for (int i = 0; i < dims.length; i++)
					dims[i] = 1;
				resultShape = new Shape(dims);
			}
		} else {
			if (!keepDimensions) {

				int j = 0;
				int[] newDims = new int[dims.length - 1];
				for (int i = 0; i < dims.length; i++) {
					if (i == axis)
						continue;
					newDims[j++] = dims[i];
				}
				resultShape = new Shape(newDims);
			} else {
				int[] newDims = this.shape.dimensions();
				newDims[axis] = 1;
				resultShape = new Shape(newDims);
			}

			int outerLoopBound = 1;
			int extremeLoopBound = 1;
			for (int i = 0; i < dims.length; i++)
				if (i < axis)
					extremeLoopBound *= dims[i];
				else if (i > axis)
					outerLoopBound *= dims[i];

			int extremeSkipper = outerLoopBound * dims[axis];
			result = new double[resultShape.total()];
			int l = 0;
			for (int k = 0; k < extremeLoopBound; k++) {
				for (int i = 0; i < outerLoopBound; i++) {
					for (int j = 0; j < dims[axis]; j++) {
						result[l] += this.data[k * extremeSkipper + j * outerLoopBound + i];
					}
					l++;
				}
			}
		}

		return new DoubleData(resultShape, result);
	}

	public boolean equals(Object obj) {

		if (!(obj instanceof DoubleData))
			return false;

		DoubleData that = (DoubleData) obj;

		if (!this.shape.equals(that.shape))
			return false;

		return Arrays.equals(this.data, that.data);
	}

	@Override
	public int hashCode() {

		return Arrays.hashCode(new int[] { this.shape.hashCode(), Arrays.hashCode(this.data) });
	}

	public DoubleData softmax(int axis) {

		return this.subtract(this.logsumexp(axis, true)).exp();
	}

	public DoubleData logsumexp(int axis, boolean keepDimension) {

		DoubleData aMax = this.amax(axis, keepDimension);
		DoubleData[] x = stretchToSameSize(this, aMax);
		aMax = x[1];
		DoubleData tmp = this.subtract(aMax).exp().sum(axis, keepDimension).log();
		x = stretchToSameSize(tmp, aMax);
		tmp = x[0];
		return tmp.add(aMax);
	}

	public DoubleData log() {
		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).lanewise(VectorOperators.LOG).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = Math.log(this.data[i]);
		}

		return new DoubleData(this.shape, result);
	}

	public DoubleData amax(Integer axis, boolean keepDimensions) {

		var dims = this.shape.dimensions();

		if (axis != null && axis >= dims.length) {
			throw new DataException("Cannot perform amax on the axis : " + axis);
		}

		double[] result;

		Shape resultShape = null;

		if (axis == null) {
			result = new double[1];
			result[0] = ArraysUtil.max(this.data);
			if (!keepDimensions) {
				resultShape = new Shape(1);
			} else {
				for (int i = 0; i < dims.length; i++)
					dims[i] = 1;
				resultShape = new Shape(dims);
			}
		} else {
			if (!keepDimensions) {

				int j = 0;
				int[] newDims = new int[dims.length - 1];
				for (int i = 0; i < dims.length; i++) {
					if (i == axis)
						continue;
					newDims[j++] = dims[i];
				}
				resultShape = new Shape(newDims);
			} else {
				int[] newDims = this.shape.dimensions();
				newDims[axis] = 1;
				resultShape = new Shape(newDims);
			}

			int outerLoopBound = 1;
			int extremeLoopBound = 1;
			for (int i = 0; i < dims.length; i++)
				if (i < axis)
					extremeLoopBound *= dims[i];
				else if (i > axis)
					outerLoopBound *= dims[i];

			int extremeSkipper = outerLoopBound * dims[axis];
			result = new double[resultShape.total()];
			int l = 0;
			int curr;
			for (int k = 0; k < extremeLoopBound; k++) {
				for (int i = 0; i < outerLoopBound; i++) {
					result[l] = this.data[k * extremeSkipper + i];
					for (int j = 1; j < dims[axis]; j++) {
						curr = k * extremeSkipper + j * outerLoopBound + i;
						if (result[l] < this.data[curr])
							result[l] = this.data[curr];
					}
					l++;
				}
			}
		}

		return new DoubleData(resultShape, result);
	}

	public DoubleData clip(double low, double high) {

		double[] d = new double[this.data.length];

		for (int i = 0; i < d.length; i++)
			if (this.data[i] < low)
				d[i] = low;
			else if (this.data[i] > high)
				d[i] = high;
			else
				d[i] = this.data[i];

		return new DoubleData(this.shape.deepCopy(), d);
	}

	public DoubleData neg() {

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;
		double[] result = new double[this.shape.total()];

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).lanewise(VectorOperators.NEG).intoArray(result, i);
		}

		for (; i < this.data.length; i++) {
			result[i] = -this.data[i];
		}

		return new DoubleData(this.shape, result);
	}

	public DoubleData unnormalize() {

		var newdims = this.shape.dimensions();

		if (newdims.length == 1 || (newdims.length == 2 && newdims[0] == 1))
			return new DoubleData(new Shape(1, 1), new double[] { this.data[0] });

		var x = this.subDataNth(0);
		newdims[0] = 1;

		return x.reShape(new Shape(newdims));
	}

	public int indexMax() {

		double m = this.data[0];
		int indm = 0;

		for (int i = 0; i < this.data.length; i++) {
			if (m >= this.data[i])
				continue;
			indm = i;
			m = this.data[i];
		}

		return indm;
	}

	public DoubleData newReShape(Shape newShape) {

		double[] newData = new double[this.data.length];
		System.arraycopy(this.data, 0, newData, 0, this.data.length);

		return new DoubleData(newShape, newData);
	}

	public DoubleData newReShape(int... dims) {
		return this.newReShape(new Shape(dims));
	}

	public static DoubleData binomial(int n, double p, Shape shape) {

		int size = shape.total();

		double[] newData = new double[size];

		double logQ = Math.log(1.0 - p);

		for (int i = 0; i < size; i++) {
			int x = 0;
			double sum = 0;
			for (;;) {
				sum += Math.log(Math.random()) / (n - x);
				if (sum < logQ) {
					newData[i] = x;
					break;
				}
				x++;
			}
		}

		return new DoubleData(shape, newData);
	}

	public DoubleData inplaceAdd(DoubleData d) {

		if (!this.shape.equals(d.getShape()))
			throw new DataException("Cannot add data of shape " + this.shape + " to " + d.shape);

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).add(DoubleVector.fromArray(SPECIES, d.data, i))
					.intoArray(this.data, i);
		}

		for (; i < this.data.length; i++) {
			this.data[i] += d.data[i];
		}

		return this;
	}

	public DoubleData inplaceMultiply(double d) {

		int upperBound = SPECIES.loopBound(this.data.length);

		int i = 0;

		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector.fromArray(SPECIES, this.data, i).mul(d).intoArray(this.data, i);
		}

		for (; i < this.data.length; i++) {
			this.data[i] = this.data[i] * d;
		}

		return this;
	}

	public DoubleData pad(int padding) {

		int[] dims = this.getShape().dimensions();
		int rows = dims[dims.length - 2];
		int columns = dims[dims.length - 1];
		int newRows = rows + padding * 2;
		int newColumns = columns + padding * 2;
		dims[dims.length - 2] = newRows;
		dims[dims.length - 1] = newColumns;

		Shape paddedShape = new Shape(dims);
		double[] d = new double[paddedShape.total()];

		int inputSkipper = rows * columns;
		int padSkipper = newRows * newColumns;

		for (int c = 0; c < this.getShape().total() / inputSkipper; c++) {

			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					d[c * padSkipper + ((i + padding) * newColumns) + j + padding] = this.data[c * inputSkipper
							+ (i * columns) + j];
				}
			}
		}

		return new DoubleData(paddedShape, d);
	}

	public DoubleData indices(Range... ranges) {

		int[] dims = this.getShape().dimensions();
		int[] sevenDims = new int[7];
		Arrays.fill(sevenDims, 1);

		for (int i = 0; i < dims.length; i++) {
			sevenDims[sevenDims.length - dims.length + i] = dims[i];
		}

		int[] skipper = new int[7];
		int total = 1;
		for (int i = sevenDims.length - 1; i >= 0; i--) {
			skipper[i] = total;
			total *= sevenDims[i];
		}

		Range[] sevenRanges = new Range[7];
		int newTotal = 1;
		for (int i = 0; i < sevenRanges.length; i++) {

			int rNum = (i - (sevenRanges.length - ranges.length));
			if (rNum >= 0 && rNum < ranges.length)
				sevenRanges[i] = ranges[rNum].setMax(sevenDims[i]);
			else
				sevenRanges[i] = new Range(0, sevenDims[i]);

			newTotal *= sevenRanges[i].total();
		}

		double[] newData = new double[newTotal];
		int cnt = 0;

		for (int i = sevenRanges[0].getFrom(); i < sevenRanges[0].getTo(); i++) {
			for (int j = sevenRanges[1].getFrom(); j < sevenRanges[1].getTo(); j++) {
				for (int k = sevenRanges[2].getFrom(); k < sevenRanges[2].getTo(); k++) {
					for (int l = sevenRanges[3].getFrom(); l < sevenRanges[3].getTo(); l++) {
						for (int m = sevenRanges[4].getFrom(); m < sevenRanges[4].getTo(); m++) {
							for (int n = sevenRanges[5].getFrom(); n < sevenRanges[5].getTo(); n++) {
								for (int o = sevenRanges[6].getFrom(); o < sevenRanges[6].getTo(); o++) {

									newData[cnt++] = this.data[i * skipper[0] + j * skipper[1] + k * skipper[2]
											+ l * skipper[3] + m * skipper[4] + n * skipper[5] + o];
								}
							}
						}
					}
				}
			}
		}

		int[] newDims = new int[dims.length];
		for (int i = sevenRanges.length - ranges.length; i < sevenRanges.length; i++) {
			newDims[i - (sevenRanges.length - ranges.length)] = sevenRanges[i].total();
		}

		return new DoubleData(new Shape(newDims), newData);
	}

	public static DoubleData generate(Shape shape) {

		double[] d = new double[shape.total()];

		for (int i = 0; i < d.length; i++) {
			d[i] = i;
		}

		return new DoubleData(shape, d);
	}
}