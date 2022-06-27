package in.dljava.functions.initializers;

import java.util.Arrays;
import java.util.Random;

import in.dljava.NotYetImplemented;
import in.dljava.data.Data;
import in.dljava.data.DoubleData;
import in.dljava.data.Shape;

public enum InitializerFunction implements Initializer {

	ZEROS {
		@Override
		public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params) {

			if (type.equals(Double.class)) {
				return new DoubleData(shape, new double[shape.total()]);
			}

			throw new NotYetImplemented();
		}
	},

	CONSTANT {
		@Override
		public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params) {

			if (params == null)
				params = new InitializerParameters();

			if (type.equals(Double.class)) {

				double[] d = new double[shape.total()];
				if (params.constant().doubleValue() != 0d)
					Arrays.fill(d, params.constant().doubleValue());
				return new DoubleData(shape, d);
			}

			throw new NotYetImplemented();
		}
	},

	ONES {

		@Override
		public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params) {

			return CONSTANT.initalize(type, shape, new InitializerParameters(1d));
		}
	},

	RANDOM {
		@Override
		public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params) {

			if (type.equals(Double.class)) {

				double[] d = new double[shape.total()];
				Random rand = new Random(params.seed());

				for (int i = 0; i < d.length; i++) {
					d[i] = rand.nextDouble();
				}

				return new DoubleData(shape, d);
			}

			throw new NotYetImplemented();
		}
	},

	RANDOM_NORMAL {
		@Override
		public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params) {

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
		}
	},

	CUSTOM {

		@Override
		public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params) {

			return params.initializer().initalize(type, shape, params);
		}
	},

	GLOROT_UNIFORM {
		@Override
		public Data initalize(Class<? extends Number> type, Shape shape, InitializerParameters params) {
			if (type.equals(Double.class)) {
//				 scale=1.0,
//		        mode='fan_avg',
//		        distribution='uniform',
//		        seed=seed

				int fan_out;
				int[] dims = shape.dimensions();
				if (dims.length == 1) {
					fan_out = dims[0];
				} else if (dims.length == 2) {
					fan_out = dims[1];
				} else {

					int f = dims[dims.length - 1] * dims[dims.length - 2];
					fan_out = dims[dims.length - 2] * f;
				}

				double scale = 1d / (fan_out < 1 ? 1d : (double) fan_out);

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
		}
	},

	;
}
