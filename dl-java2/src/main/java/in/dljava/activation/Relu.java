package in.dljava.activation;

public class Relu implements ActivationFunction {

	@Override
	public void apply(double[] d) {
		for (int i = 0; i < d.length; i++)
			d[i] = d[i] < 0d ? 0d : d[i];
	}

	@Override
	public void applyDerivative(double[] d) {

		for (int i = 0; i < d.length; i++)
			d[i] = d[i] <= 0d ? 0d : 1d;

	}

}
