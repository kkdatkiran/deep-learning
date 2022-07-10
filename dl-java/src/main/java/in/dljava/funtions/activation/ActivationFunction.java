package in.dljava.funtions.activation;

import in.dljava.util.BoundUtil;

public enum ActivationFunction implements ActivationMaker {

	LINEAR {
		@Override
		public Activation make() {

			return new Activation() {

				@Override
				public void apply(double[] v) {
					// Nothing need to be done
				}

				@Override
				public double[] derivative(double[] v) {
					double[] d = new double[v.length];

					for (int i = 0; i < v.length; i++)
						d[i] = 1d;

					return d;
				}

			};
		}
	},
	RELU {
		@Override
		public Activation make() {
			return new Activation() {

				@Override
				public void apply(double[] v) {

					for (int i = 0; i < v.length; i++)
						v[i] = v[i] < 0d ? 0d : v[i];
				}

				@Override
				public double[] derivative(double[] v) {

					double[] d = new double[v.length];

					for (int i = 0; i < v.length; i++)
						d[i] = v[i] <= 0d ? 0d : 1d;

					return d;
				}

			};
		}
	},
	SIGMOID {
		@Override
		public Activation make() {

			return new Activation() {

				@Override
				public double[] derivative(double[] v) {
					double[] d = new double[v.length];

					for (int i = 0; i < v.length; i++)
						d[i] = v[i] * (1d - v[i]);

					return d;
				}

				@Override
				public void apply(double[] v) {
					for (int i = 0; i < v.length; i++)
						v[i] = 1 / (1 + BoundUtil.bound(Math.exp(-1 * v[i])));

				}
			};
		}
	};
}
