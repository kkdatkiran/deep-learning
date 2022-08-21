package in.dljava.data;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import org.junit.jupiter.api.Test;

class NumberGeneratorTest {

	@Test
	void testNextNumber() {

		NumberGenerator ng = new NumberGenerator(new int[] { 4, 6 }, new int[] { 1, 0 });

		List<Integer> sequence = new ArrayList<>();
		int n;
		int max = -1;
		int min = Integer.MAX_VALUE;
		while ((n = ng.nextNumber()) != -1) {
			sequence.add(n);
			if (max < n)
				max = n;
			if (min > n)
				min = n;
		}

		int total = 24;

		assertEquals(total, (new HashSet<>(sequence)).size());
		assertEquals(total - 1, max);
		assertEquals(0, min);

		System.out.println(sequence);
	}

	@Test
	void testNextNumber1() {

		NumberGenerator ng = new NumberGenerator(new int[] { 2, 2 }, new int[] { 1, 0 });

		List<Integer> sequence = new ArrayList<>();
		int n;
		int max = -1;
		int min = Integer.MAX_VALUE;
		while ((n = ng.nextNumber()) != -1) {
			sequence.add(n);
			if (max < n)
				max = n;
			if (min > n)
				min = n;
		}

		int total = 4;

		assertEquals(total, (new HashSet<>(sequence)).size());
		assertEquals(total - 1, max);
		assertEquals(0, min);

		System.out.println(sequence);
	}

	@Test
	void testNextNumber2() {

		NumberGenerator ng = new NumberGenerator(new int[] { 2, 1, 3 }, new int[] { 1, 2, 0 });

		List<Integer> sequence = new ArrayList<>();
		int n;
		int max = -1;
		int min = Integer.MAX_VALUE;
		while ((n = ng.nextNumber()) != -1) {
			sequence.add(n);
			if (max < n)
				max = n;
			if (min > n)
				min = n;
		}
		System.out.println(sequence);

		int total = 6;

		assertEquals(total, (new HashSet<>(sequence)).size());
		assertEquals(total - 1, max);
		assertEquals(0, min);
	}

	@Test
	void testNextNumber3() {
		NumberGenerator ng = new NumberGenerator(new int[] { 5, 4, 3, 2 }, new int[] { 2, 3, 0, 1 });

		List<Integer> sequence = new ArrayList<>();
		int n;
		int max = -1;
		int min = Integer.MAX_VALUE;
		while ((n = ng.nextNumber()) != -1) {
			sequence.add(n);
			if (max < n)
				max = n;
			if (min > n)
				min = n;
		}
		System.out.println(sequence);

		int total = 120;

		assertEquals(total, (new HashSet<>(sequence)).size());
		assertEquals(total - 1, max);
		assertEquals(0, min);
	}

}
