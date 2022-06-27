package in.dljava.data;

import java.util.Arrays;

public class DoubleData implements Data {

	private double[] data;
	private Shape shape;

	public DoubleData(Shape shape, double[] data) {

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
}
