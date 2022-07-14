package in.dljava.optimizer;

import lombok.Data;

@Data
public class Tuple2<F,S> {

	private F t1;
	private S t2;

	public Tuple2(F f, S s) {
		this.t1 = f;
		this.t2 = s;
	}
}
