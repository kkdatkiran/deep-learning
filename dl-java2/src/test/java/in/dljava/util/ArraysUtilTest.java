package in.dljava.util;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

class ArraysUtilTest {

	@Test
	void testSplice() {

		assertArrayEquals(new int[] { 1, 2, 5, 6, 3, 4 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 2, 0, 5, 6));
		assertArrayEquals(new int[] { 1, 2, 3, 4 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 2, 0));
		assertArrayEquals(new int[] { 4 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 0, 3));
		assertArrayEquals(new int[] { 1, 2, 3, 4 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 4, 0));
		assertArrayEquals(new int[] { 1, 2, 3, 12, 4 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 3, 0, 12));
		assertArrayEquals(new int[] { 1, 2, 3, 4, 12 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 4, 0, 12));
		assertArrayEquals(new int[] { 1, 2, 3 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 3, 1));
		assertArrayEquals(new int[] { 1, 2, 4 }, ArraysUtil.splice(new int[] { 1, 2, 3, 4 }, 2, 1));
	}

}
