package in.dljava.operations;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;

class SoftmaxTest {

	@Test
	void test() {
		Softmax smax = new Softmax();
		DoubleData d = smax.forward(new DoubleData(new Shape(1, 2), new double[] { 1, 2 }), false);
		assertArrayEquals(new double[] { 0.2689414213699951, 0.7310585786300049 }, d.getData());
		smax.backward(new DoubleData(new Shape(1, 2), new double[] { 0.2234233, 0.323554234 })).print();
	}

}