package in.dljava.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class DoubleDataTest {

	@Test
	void testTranspose() {
		DoubleData d = new DoubleData(new Shape(3, 2), new double[] { 1, 2, 3, 4, 5, 6 });
		DoubleData t = new DoubleData(new Shape(2, 3), new double[] { 1, 3, 5, 2, 4, 6 });

		assertArrayEquals(t.getData(), d.transpose().getData());
	}

	@Test
	void testSum2d() {

		DoubleData d = new DoubleData(new Shape(6, 3),
				new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

		DoubleData nullAxis = new DoubleData(new Shape(1), new double[] { 90 });

		assertEquals(nullAxis, d.sum(null, false));
		assertEquals(nullAxis.reShape(new Shape(1, 1)), d.sum(null, true));

		DoubleData zeroAxis = new DoubleData(new Shape(3), new double[] { 24, 30, 36 });

		assertEquals(zeroAxis, d.sum(0, false));
		assertEquals(zeroAxis.reShape(new Shape(1, 3)), d.sum(0, true));

		DoubleData oneAxis = new DoubleData(new Shape(6), new double[] { 6, 15, 24, 6, 15, 24 });

		assertEquals(oneAxis, d.sum(1, false));
		assertEquals(oneAxis.reShape(new Shape(6, 1)), d.sum(1, true));
	}

	@Test
	void testSum3d() {

		DoubleData d = new DoubleData(new Shape(2, 6, 3), new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7,
				8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

		DoubleData nullAxis = new DoubleData(new Shape(1), new double[] { 180 });

		assertEquals(nullAxis, d.sum(null, false));
		assertEquals(nullAxis.reShape(new Shape(1, 1, 1)), d.sum(null, true));

		DoubleData zeroAxis = new DoubleData(new Shape(6, 3),
				new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 2, 4, 6, 8, 10, 12, 14, 16, 18 });

		assertEquals(zeroAxis, d.sum(0, false));
		assertEquals(zeroAxis.reShape(new Shape(1, 6, 3)), d.sum(0, true));

		DoubleData oneAxis = new DoubleData(new Shape(2, 3), new double[] { 24, 30, 36, 24, 30, 36 });

		assertEquals(oneAxis, d.sum(1, false));
		assertEquals(oneAxis.reShape(new Shape(2, 1, 3)), d.sum(1, true));

		DoubleData twoAxis = new DoubleData(new Shape(2, 6),
				new double[] { 6, 15, 24, 6, 15, 24, 6, 15, 24, 6, 15, 24 });

		assertEquals(twoAxis, d.sum(2, false));
		assertEquals(twoAxis.reShape(new Shape(2, 6, 1)), d.sum(2, true));
	}

	@Test
	void testMultiply() {

		DoubleData a = new DoubleData(new Shape(3, 2), new double[] { 1, 2, 3, 4, 5, 6 });

		DoubleData b = new DoubleData(new Shape(3, 2), new double[] { 1, 3, 5, 2, 4, 6 });
		DoubleData result = new DoubleData(new Shape(3, 2), new double[] { 1, 6, 15, 8, 20, 36 });
		assertEquals(result, a.multiply(b));

		a = new DoubleData(new Shape(2, 3), new double[] { 1, 2, 3, 4, 5, 6 });
		b = new DoubleData(new Shape(2, 1), new double[] { 2, 3 });
		result = new DoubleData(new Shape(2, 3), new double[] { 2, 4, 6, 12, 15, 18 });
		assertEquals(result, a.multiply(b));
	}

	@Test
	void testDivision() {
		DoubleData a = new DoubleData(new Shape(2, 3), new double[] { 1, 2, 3, 4, 5, 6 });
		DoubleData b = new DoubleData(new Shape(2, 1), new double[] { 2, 3 });
		DoubleData result = new DoubleData(new Shape(2, 3),
				new double[] { 0.5, 1, 1.5, 1.3333333333333333, 1.6666666666666667, 2 });

		assertEquals(result, a.divide(b));
	}

	@Test
	void testStretch() {

		DoubleData a = new DoubleData(new Shape(1, 1, 2), new double[] { 1, 2 });
		DoubleData x = a.stretchDimension(1, 2);
		assertEquals(new DoubleData(new Shape(1, 2, 2), new double[] { 1, 2, 1, 2 }), x);

		DoubleData y = x.stretchDimension(2, 3);
		assertEquals(new DoubleData(new Shape(3, 2, 2), new double[] { 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 }), y);

		DoubleData b = new DoubleData(new Shape(3, 1, 1), new double[] { 3, 2, 1 });
		x = b.stretchDimension(0, 3);
		assertEquals(new DoubleData(new Shape(3, 1, 3), new double[] { 3, 3, 3, 2, 2, 2, 1, 1, 1 }), x);
		y = x.stretchDimension(1, 2);
		assertEquals(new DoubleData(new Shape(3, 2, 3),
				new double[] { 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1 }), y);

	}

	@Test
	void testConcatenate() {

		DoubleData a = new DoubleData(new Shape(2, 2), new double[] { 1, 2, 3, 4 });
		DoubleData b = new DoubleData(new Shape(1, 2), new double[] { 5, 6 });

		assertEquals(new DoubleData(new Shape(3, 2), new double[] { 1, 2, 3, 4, 5, 6 }), a.concatenate(b, 0));
		assertEquals(new DoubleData(new Shape(6), new double[] { 1, 2, 3, 4, 5, 6 }), a.concatenate(b, null));
		assertEquals(new DoubleData(new Shape(2, 3), new double[] { 1, 2, 5, 3, 4, 6 }),
				a.concatenate(b.transpose(), 1));
	}

	@Test
	void testAMax() {

		DoubleData a = new DoubleData(new Shape(2, 2), new double[] { 0, 1, 2, 3 });

		assertEquals(new DoubleData(new Shape(2), new double[] { 2, 3 }), a.amax(0, false));
		assertEquals(new DoubleData(new Shape(1, 2), new double[] { 2, 3 }), a.amax(0, true));
		assertEquals(new DoubleData(new Shape(1), new double[] { 3 }), a.amax(null, false));
	}

	@Test
	void testUnNormalize() {

		DoubleData a = new DoubleData(new Shape(2, 2), new double[] { 0, 1, 2, 3 });
		assertEquals(new DoubleData(new Shape(1, 2), new double[] { 0, 1 }), a.unnormalize());

		a = new DoubleData(new Shape(3, 2, 2), new double[] { 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 });

		assertEquals(new DoubleData(new Shape(1, 2, 2), new double[] { 1, 2, 1, 2 }), a.unnormalize());
	}

	@Test
	void testTranspose2() {

		DoubleData a = new DoubleData(new Shape(2, 4), new double[] { 0, 1, 2, 3, 4, 5, 6, 7 });
		assertEquals(a.transpose(), a.transpose(new int[] { 1, 0 }));
	}

	@Test
	void testTranspose3() {

		double[] d = new double[120];
		for (int i = 0; i < d.length; i++)
			d[i] = i;

		DoubleData a = new DoubleData(new Shape(2, 3, 4, 5), d);

		assertArrayEquals(new double[] { 0, 20, 40, 60, 80, 100, 1, 21, 41, 61, 81, 101, 2, 22, 42, 62, 82, 102, 3, 23,
				43, 63, 83, 103, 4, 24, 44, 64, 84, 104, 5, 25, 45, 65, 85, 105, 6, 26, 46, 66, 86, 106, 7, 27, 47, 67,
				87, 107, 8, 28, 48, 68, 88, 108, 9, 29, 49, 69, 89, 109, 10, 30, 50, 70, 90, 110, 11, 31, 51, 71, 91,
				111, 12, 32, 52, 72, 92, 112, 13, 33, 53, 73, 93, 113, 14, 34, 54, 74, 94, 114, 15, 35, 55, 75, 95, 115,
				16, 36, 56, 76, 96, 116, 17, 37, 57, 77, 97, 117, 18, 38, 58, 78, 98, 118, 19, 39, 59, 79, 99, 119 },
				a.transpose(new int[] { 2, 3, 0, 1 }).getData());
	}
	
}
