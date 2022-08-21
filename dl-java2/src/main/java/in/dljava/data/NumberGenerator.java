package in.dljava.data;

public class NumberGenerator {

	private int[] sizeOfEachAxes;
	private int[] axes;
	private int[] counter;
	private int nAxes;
	private int[] axesTotals;

	public NumberGenerator(int[] sizeOfEachAxes, int[] axes) {

		this.sizeOfEachAxes = sizeOfEachAxes;
		this.axes = axes;
		this.counter = new int[this.sizeOfEachAxes.length];
		this.nAxes = this.axes.length;
		this.axesTotals = new int[this.sizeOfEachAxes.length + 1];

		this.axesTotals[0] = 1;

		for (int i = 1; i < this.axesTotals.length; i++) {
			this.axesTotals[i] = this.axesTotals[i - 1] * this.sizeOfEachAxes[i - 1];
		}

//		System.out.println(Arrays.toString(this.axesTotals));
	}

	public int nextNumber() {

		int total = 0;

//		System.out.println(Arrays.toString(this.counter));
		for (int i = 0; i < this.axes.length; i++) {
			total += this.counter[i] * this.axesTotals[this.axes[i]];
		}

		if (this.counter[this.nAxes - 1] == this.sizeOfEachAxes[this.axes[this.nAxes - 1]])
			return -1;

		this.counter[0]++;

		for (int i = 0; i < this.axes.length; i++) {
			if (this.counter[i] >= this.sizeOfEachAxes[this.axes[i]] && (i + 1 != this.nAxes)) {
				this.counter[i] = 0;
				this.counter[i + 1]++;
			}
		}

		return total;
	}
}