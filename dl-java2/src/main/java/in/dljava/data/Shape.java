package in.dljava.data;

import java.util.Arrays;

public class Shape {

	private final int[] dimensions;
	private int totalBackup = -1;

	public Shape(int... dimensions) {

		if (dimensions == null || dimensions.length == 0)
			throw new DataException("Data shape cannot be of no dimension");

		this.dimensions = dimensions;
		for (int each : this.dimensions)
			if (each < 0)
				throw new DataException("Negative dimensions are not allowed");
	}

	public Shape increaseOneDimension() {
		int[] subDimension = new int[dimensions.length + 1];

		System.arraycopy(dimensions, 0, subDimension, 1, dimensions.length);
		subDimension[0] = 1;

		return new Shape(subDimension);
	}

	public Shape oneOfHigherDimension() {

		int[] subDimension = new int[dimensions.length];

		System.arraycopy(dimensions, 0, subDimension, 0, dimensions.length);
		subDimension[0] = 1;

		return new Shape(subDimension);
	}

	public int dimensions(int i) {
		return this.dimensions[i];
	}

	public int[] dimensions() {

		int[] x = new int[dimensions.length];

		System.arraycopy(dimensions, 0, x, 0, dimensions.length);
		return x;
	}

	public int total() {

		if (totalBackup > 0)
			return this.totalBackup;

		int total = 1;

		for (int e : this.dimensions)
			total *= e;

		this.totalBackup = total;

		return total;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof Shape s) {
			if (s.dimensions.length != this.dimensions.length)
				return false;

			for (int i = 0; i < s.dimensions.length; i++)
				if (s.dimensions[i] != this.dimensions[i])
					return false;

			return true;
		}

		return false;
	}

	@Override
	public int hashCode() {
		return Arrays.hashCode(this.dimensions);
	}

	@Override
	public String toString() {

		StringBuilder sb = new StringBuilder("Shape : [");

		for (int i : this.dimensions)
			sb.append(i).append(", ");

		return sb.delete(sb.length() - 2, sb.length()).append("]").toString();
	}

	public int totalInAxis(int axis) {

		if (axis == 0)
			return this.total();

		if (axis < 0 || axis >= this.dimensions.length)
			throw new DataException("Data shape don't have enough axes");

		int total = 1;
		for (int i = axis; i < this.dimensions.length; i++) {
			total *= this.dimensions[i];
		}

		return total;
	}

	public Shape oneOf() {

		int[] dims = new int[this.dimensions.length];

		System.arraycopy(this.dimensions, 0, dims, 0, dims.length);
		dims[0] = 1;

		return new Shape(dims);
	}

	public Shape oneOfLooseFirstDim() {

		int[] dims = new int[this.dimensions.length - 1];

		System.arraycopy(this.dimensions, 1, dims, 0, dims.length);

		return new Shape(dims);
	}

	public Shape deepCopy() {

		return new Shape(Arrays.copyOf(this.dimensions, this.dimensions.length));
	}

	public int numberOfAxes() {

		return this.dimensions.length;
	}

	public int[] totalsInAxes() {

		int[] totals = new int[this.dimensions.length + 1];
		totals[this.dimensions.length] = 1;

		for (int i = this.dimensions.length - 1; i >= 0; i--) {

			totals[i] = totals[i + 1] * this.dimensions[i];
		}

		return totals;
	}
}
