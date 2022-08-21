package in.dljava.operations;

import in.dljava.data.DoubleData;

public class Flatten extends Operation {

	@Override
	public DoubleData output(boolean inference) {

		int fd = this.input.getShape().dimensions()[0];
		return this.input.newReShape(fd, this.input.getShape().total() / fd);
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {
		return outGradient.newReShape(this.input.getShape());
	}

	@Override
	public Flatten deepCopy() {
		Flatten f = new Flatten();
		f.inpGradient = this.inpGradient.deepCopy();
		f.input = this.input.deepCopy();
		f.out = this.out.deepCopy();
		return f;
	}

}
