package in.dljava.data;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class DoubleDataTest {

	@Test
	void test() {

		DoubleData a = new DoubleData(new Shape(2, 3), new double[] { 1, 2, 3, 4, 5, 6 });
		DoubleData b = new DoubleData(new Shape(3, 2), new double[] { 10, 11, 20, 21, 30, 31 });

		DoubleData c = a.matrixMultiply(b);

		assertArrayEquals(new double[] { 140d, 146d, 320d, 335d }, c.getData());
	}

}
