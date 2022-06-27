package in.dljava.layers;

import in.dljava.data.Shape;

public class Input implements Layer {

	private final Shape shape;

	public Input(Shape shape) {

		this.shape = shape;
	}

	public Shape getShape() {
		return this.shape;
	}
}
