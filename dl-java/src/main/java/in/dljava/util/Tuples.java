package in.dljava.util;

public class Tuples {

	public static <F, S> Tuple2<F, S> of(F t1, S t2) {
		return new Tuple2<>(t1, t2);
	}

	public static <F, S, T> Tuple3<F, S, T> of(F t1, S t2, T t3) {
		return new Tuple3<>(t1, t2, t3);
	}

	private Tuples() {
	}
}
