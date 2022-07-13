package in.dljava.activation;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

class SigmoidTest {

	@Test
	void testApply() {

		double[] d = new double[100];
		
		for (int i = 0; i < d.length; i++)
			d[i] = Math.random();
		
		double[] sig = new double[d.length];
		
		for (int i=0; i < d.length; i++) {
			sig[i] = 1d / (1d + Math.exp(-d[i]));
		}
	
		new Sigmoid().apply(d);
		
		assertArrayEquals(sig, d);
		
	}

	@Test
	void testApplyDerivative() {
		
		double[] d = new double[100];
		
		for (int i = 0; i < d.length; i++)
			d[i] = Math.random();
		
		double[] sig = new double[d.length];
		
		for (int i=0; i < d.length; i++) {
			sig[i] = d[i] * (1 - d[i]);
		}
	
		new Sigmoid().applyDerivative(d);
		
		assertArrayEquals(sig, d);
	}

}
