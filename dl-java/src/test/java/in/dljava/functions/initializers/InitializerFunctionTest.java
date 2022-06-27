package in.dljava.functions.initializers;

import org.junit.jupiter.api.Test;

import in.dljava.data.Data;
import in.dljava.data.Shape;

class InitializerFunctionTest {

	@Test
	void test() {
		
		Data d = InitializerFunction.GLOROT_UNIFORM.initalize(Double.class, new Shape(100), new InitializerParameters());
		System.out.println(d);
	}

}
