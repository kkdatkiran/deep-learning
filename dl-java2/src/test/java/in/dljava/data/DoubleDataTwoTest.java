package in.dljava.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

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

	@Test
	void testPadding() {

		DoubleData d = new DoubleData(new Shape(2, 3, 2, 2),
				new double[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });

		assertEquals(new DoubleData(new Shape(2, 3, 6, 6),
				new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0,
						0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 3, 4,
						0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0,
						0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
						0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }),
				d.pad(2));
	}

	@Test
	void testIndices() {
		DoubleData d = DoubleData.generate(new Shape(3, 3, 3));
		d.print();
		d.indices(new Range(-1), new Range(-1), new Range(1, 3)).print();;
	}
}