package in.dljava.data;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class DoubleDataTest {

	@Test
	void testTranspose() {
		DoubleData d = new DoubleData(new Shape(3,2), new double[] {1,2,3,4,5,6});
		DoubleData t = new DoubleData(new Shape(2,3), new double[] {1,3,5,2,4,6});
		
		assertArrayEquals(t.getData(), d.transpose().getData());
	}

}
