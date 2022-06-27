package in.dljava.functions.initializers;

public record InitializerParameters(Number constant, Long seed, int axis, Initializer initializer) {

	public InitializerParameters() {
		this(null, null, 0, null);
	}

	public InitializerParameters(Number constant) {
		this(constant, null, 0, null);
	}

	public InitializerParameters(Long seed) {
		this(null, seed, 0, null);
	}

	public InitializerParameters(Long seed, int axis) {
		this(null, seed, axis, null);
	}

	public InitializerParameters(Initializer initializer) {
		this(null, null, 0, initializer);
	}

	public InitializerParameters(Number constant, Long seed, int axis, Initializer initializer) {

		this.constant = constant == null ? 0d : constant;

		this.seed = seed == null ? System.currentTimeMillis() : seed;

		this.initializer = initializer;

		this.axis = axis;
	}
}
