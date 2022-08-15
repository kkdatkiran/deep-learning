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
}
