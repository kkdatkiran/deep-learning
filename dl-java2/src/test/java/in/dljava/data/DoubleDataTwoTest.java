package in.dljava.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class DoubleDataTwoTest {

	@Test
	void testMatrixMultiplyMD() {

		DoubleData a = DoubleData.full(new Shape(4, 3, 3, 3), 1);
		DoubleData b = DoubleData.full(new Shape(3, 2), 2);

		Shape finShape = new Shape(4, 3, 3, 2);
		double d[] = new double[finShape.total()];
		
		Arrays.fill(d, 6);

		assertArrayEquals(d, a.matrixMultiplyND(b).getData());
	}
}