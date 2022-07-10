package in.dljava.functions.initializers;

import org.junit.jupiter.api.Test;

import in.dljava.data.Data;
import in.dljava.data.Shape;

class InitializerFunctionTest {

	@Test
	void test() {

		Data d = InitializerFunction.GLOROT_UNIFORM.make(Double.class, new InitializerParameters())
				.initalize(new Shape(1000));
		System.out.println(d);
	}

}
