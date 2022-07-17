package in.dljava.optimizer;

import lombok.Data;

@Data
public class Tuple4<F, S, T, R> {

	private F t1;
	private S t2;
	private T t3;
	private R t4;
	
	public Tuple4() {
	}

	public Tuple4(F f, S s, T t, R r) {
		this.t1 = f;
		this.t2 = s;
		this.t3 = t;
		this.t4 = r;
	}
}
