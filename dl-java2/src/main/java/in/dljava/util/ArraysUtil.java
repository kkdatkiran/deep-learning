package in.dljava.util;

public class ArraysUtil {

	public static double max(final double[] d, int from, int to, int step) {

		if (from > to) {
			int x = from;
			from = to;
			to = x;
			step *= -1;
		}

		double max = d[from++];

		while (from < to) {
			if (max < d[from])
				max = d[from];
			from += step;
		}

		return max;
	}

	public static double max(final double[] d) {
		return max(d, 0, d.length, 1);
	}

	private ArraysUtil() {

	}
}
