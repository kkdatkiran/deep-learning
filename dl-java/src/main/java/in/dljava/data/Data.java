package in.dljava.data;

public interface Data {

	public abstract Shape getShape();

	public static Data of(Shape s, double[] data) {
		return new DoubleData(s, data);
	}

	public static Data of(double[] data) {

		if (data == null)
			throw new DataException("Cannot create Data for null value");

		return of(new Shape(data.length), data);
	}

	public static Data of(double[][] data) {

		if (data == null || data.length < 1)
			throw new DataException("Cannot create Data for null value or no data");

		double[] dataReshaped = new double[data.length * data[0].length];

		for (int i = 0; i < data.length; i++) {

			if (data[i].length != data[0].length)
				throw new DataException("Data is has variable column size rows");

			System.arraycopy(data[i], 0, dataReshaped, i * data[0].length, data[0].length);
		}

		return new DoubleData(new Shape(data.length, data[0].length), dataReshaped);
	}
}
