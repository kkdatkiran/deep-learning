package in.dljava.data;

public class Range {

	private int from;
	private int to;

	public Range(int n) {
		this.from = 0;
		this.to = n;
	}

	public Range(int from, int to) {
		this.from = from;
		this.to = to;
	}

	public int getFrom(int max) {
		if (from < max)
			return from;
		return max;
	}

	public int getTo(int max) {

		if (to == -1)
			return max;

		if (to <= max)
			return to;

		return max;
	}

	public int getFrom() {
		return this.from;
	}

	public int getTo() {
		return this.to;
	}

	public Range setMax(int max) {

		this.from = getFrom(max);
		this.to = getTo(max);

		return this;
	}

	public int total() {
		return this.to - this.from;
	}

	@Override
	public String toString() {
		return this.from + " -  " + this.to;
	}
}
