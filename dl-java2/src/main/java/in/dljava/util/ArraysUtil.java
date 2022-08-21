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

	public static int[] inplaceReverse(final int[] src) {

		int i = 0;
		int j = src.length - 1;
		int temp;

		while (i < j) {
			temp = src[i];
			src[i] = src[j];
			src[j] = temp;
			i++;
			j--;
		}

		return src;
	}

	private ArraysUtil() {

	}
}
