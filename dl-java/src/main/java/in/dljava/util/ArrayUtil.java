package in.dljava.util;

import in.dljava.DLException;

public class ArrayUtil {

	public static Object[][] zip(Object[] first, Object[] second) {

		if (first == null || second == null || first.length != second.length)
			throw new DLException("Cannot zip the provided two arrays");

		Object[][] result = new Object[first.length][2];

		for (int i = 0; i < first.length; i++) {

			result[i][0] = first[i];
			result[i][1] = second[i];
		}

		return result;
	}

	public static String[] zipToString(Object[] first, Object[] second) {

		if (first == null || second == null || first.length != second.length)
			throw new DLException("Cannot zip the provided two arrays");

		String[] result = new String[first.length];

		for (int i = 0; i < first.length; i++) {

			result[i] = "[" + first[i] + ", " + second[i] + "]";
		}

		return result;
	}
	
	public static String[] zipToString(Object[] first, double[] second) {

		if (first == null || second == null || first.length != second.length)
			throw new DLException("Cannot zip the provided two arrays");

		String[] result = new String[first.length];

		for (int i = 0; i < first.length; i++) {

			result[i] = "[" + first[i] + ", " + second[i] + "]";
		}

		return result;
	}

	private ArrayUtil() {
	}
}
