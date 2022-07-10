package in.dljava.util;

public class StringUtil {

	public static String padEnding(String str, int length) {

		if (str.length() >= length)
			return str.substring(0, length);

		StringBuilder sb = new StringBuilder(str);

		while (sb.length() < length)
			sb.append(" ");

		return sb.toString();
	}

	private StringUtil() {

	}
}
