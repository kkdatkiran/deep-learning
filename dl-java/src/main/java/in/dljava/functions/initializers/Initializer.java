package in.dljava.functions.initializers;

import in.dljava.data.Data;
import in.dljava.data.Shape;

public interface Initializer {

	public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params);
}
