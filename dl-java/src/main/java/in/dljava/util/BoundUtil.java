package in.dljava.util;

public class BoundUtil {

	public static double bound(double v) {
		if (v < -1.0e20)
			return -1.0e20;
		else if (v > 1.0e20)
			return 1.0e20;
		return v;
	}

	private BoundUtil() {
	}
}
