package in.dljava.functions.initializers;

public interface InitializerMaker {

	public Initializer make(Class<? extends Number> type,InitializerParameters params);

	public default Initializer make(Class<? extends Number> type) {
		return this.make(type, new InitializerParameters());
	}
}
