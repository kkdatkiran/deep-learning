package in.dljava.data;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ShapeTest {

	@Test
	void testTotalInAxes() {

		Shape s = new Shape(3, 2, 4);

		assertArrayEquals(new int[] { 24, 8, 4 , 1}, s.totalsInAxes());
	}

}
