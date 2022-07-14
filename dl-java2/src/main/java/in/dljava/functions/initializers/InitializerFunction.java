package in.dljava.functions.initializers;

import java.util.Arrays;
import java.util.Random;

import in.dljava.NotYetImplemented;
import in.dljava.data.DoubleData;

public enum InitializerFunction implements InitializerMaker {

	ZEROS {

		@Override
		public Initializer make(final Class<? extends Number> type, final InitializerParameters params) {

			return shape -> {

				if (type.equals(Double.class)) {
					return new DoubleData(shape, new double[shape.total()]);
				}

				throw new NotYetImplemented();
			};
		}

	},

	CONSTANT {
		@Override
		public Initializer make(Class<? extends Number> type, InitializerParameters nparams) {

			return shape -> {

				InitializerParameters params = nparams;
				if (params == null)
					params = new InitializerParameters();

				if (type.equals(Double.class)) {

					double[] d = new double[shape.total()];
					if (params.constant().doubleValue() != 0d)
						Arrays.fill(d, params.constant().doubleValue());
					return new DoubleData(shape, d);
				}

				throw new NotYetImplemented();
			};
		}
	},

	ONES {

		@Override
		public Initializer make(Class<? extends Number> type, InitializerParameters params) {

			return CONSTANT.make(type, new InitializerParameters(1d));
		}
	},

	RANDOM {
		@Override
		public Initializer make(Class<? extends Number> type, InitializerParameters params) {
			return shape -> {
				if (type.equals(Double.class)) {

					double[] d = new double[shape.total()];
					Random rand = new Random(params.seed());

					for (int i = 0; i < d.length; i++) {
						d[i] = rand.nextDouble();
					}

					return new DoubleData(shape, d);
				}

				throw new NotYetImplemented();
			};
		}
	},

	RANDOM_NORMAL {
		@Override
		public Initializer make(Class<? extends Number> type, InitializerParameters params) {
			return shape -> {
				if (type.equals(Double.class)) {

					double[] d = new double[shape.total()];
					Random rand = new Random(params.seed());

					final int groupSize = shape.totalInAxis(params.axis());
					int count = groupSize;

					for (int i = 0; i < d.length; i++) {
						d[i] = rand.nextGaussian();
						count--;

						if (count == 0) {
							rand = new Random(params.seed());
							count = groupSize;
						}
					}

					return new DoubleData(shape, d);
				}

				throw new NotYetImplemented();
			};
		}
	},

	CUSTOM {

		@Override
		public Initializer make(Class<? extends Number> type, InitializerParameters params) {
			return params.initializer();
		}
	},

	GLOROT_UNIFORM {
		@Override
		public Initializer make(Class<? extends Number> type, InitializerParameters params) {
			return shape -> {
				if (type.equals(Double.class)) {

					int fanOut;
					int[] dims = shape.dimensions();
					if (dims.length == 1) {
						fanOut = dims[0];
					} else if (dims.length == 2) {
						fanOut = dims[1];
					} else {

						int f = dims[dims.length - 1] * dims[dims.length - 2];
						fanOut = dims[dims.length - 2] * f;
					}

					double scale = 1d / (fanOut < 1 ? 1d : (double) fanOut);

					double trunc = 2d * (Math.sqrt(scale) / .87962566103423978d);

					double[] d = new double[shape.total()];
					Random rand = new Random(params.seed());

					final int groupSize = shape.totalInAxis(params.axis());
					int count = groupSize;

					for (int i = 0; i < d.length; i++) {
						do {
							d[i] = rand.nextGaussian();
						} while (d[i] < -trunc || d[i] > trunc);
						count--;

						if (count == 0) {
							rand = new Random(params.seed());
							count = groupSize;
						}
					}

					return new DoubleData(shape, d);

				}

				throw new NotYetImplemented();
			};
		}
	},

	;
}
