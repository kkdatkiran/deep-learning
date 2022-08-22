package in.dljava.optimizer;

import lombok.Data;

@Data
public class Tuple3<F, S, T> {

	private F t1;
	private S t2;
	private T t3;

	public Tuple3() {
	}

	public Tuple3(F f, S s, T t) {
		this.t1 = f;
		this.t2 = s;
		this.t3 = t;
	}
}
