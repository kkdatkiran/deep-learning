package in.dljava.functions.initializers;

import in.dljava.data.Data;
import in.dljava.data.Shape;

@FunctionalInterface
public interface Initializer {

	public Data initalize(Shape s);
}
