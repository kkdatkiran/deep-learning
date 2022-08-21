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

	public static int[] splice(int[] src, int start, int count, int... nums) {

		int[] newArray = new int[src.length - count + (nums == null ? 0 : nums.length)];

		int i = 0;
		int j = 0;

		while (i < newArray.length) {

			if (j == start) {
				j += count;
				int k = 0;
				while (nums != null && k < nums.length) {
					newArray[i++] = nums[k++];
				}
			}

			if (i < newArray.length && j < src.length)
				newArray[i++] = src[j++];
		}

		return newArray;
	}
}
