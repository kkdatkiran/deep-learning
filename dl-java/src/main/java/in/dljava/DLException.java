package in.dljava;

public class DLException extends RuntimeException {

	private static final long serialVersionUID = 8236188769125669786L;

	public DLException(String msg) {
		super(msg);
	}

	public DLException(String msg, Throwable ex) {
		super(msg, ex);
	}
}
