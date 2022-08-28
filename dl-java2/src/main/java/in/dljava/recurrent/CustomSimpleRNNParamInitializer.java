package in.dljava.recurrent;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

import lombok.val;

public class CustomSimpleRNNParamInitializer implements ParamInitializer {

	private static final CustomSimpleRNNParamInitializer INSTANCE = new CustomSimpleRNNParamInitializer();

	public static CustomSimpleRNNParamInitializer getInstance() {
		return INSTANCE;
	}

	public static final String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
	public static final String RECURRENT_WEIGHT_KEY = "RW";
	public static final String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
	public static final String GAIN_KEY = DefaultParamInitializer.GAIN_KEY;

	private static final List<String> WEIGHT_KEYS = Collections
			.unmodifiableList(Arrays.asList(WEIGHT_KEY, RECURRENT_WEIGHT_KEY));
	private static final List<String> BIAS_KEYS = Collections.singletonList(BIAS_KEY);

	@Override
	public long numParams(NeuralNetConfiguration conf) {
		return numParams(conf.getLayer());
	}

	@Override
	public long numParams(Layer layer) {
		CustomSimpleRNNConf c = (CustomSimpleRNNConf) layer;
		val nIn = c.getNIn();
		val nOut = c.getNOut();
		return nIn * nOut + nOut * nOut + nOut + (hasLayerNorm(layer) ? 2 * nOut : 0);
	}

	@Override
	public List<String> paramKeys(Layer layer) {
		final ArrayList<String> keys = new ArrayList<>(3);
		keys.addAll(weightKeys(layer));
		keys.addAll(biasKeys(layer));
		return keys;
	}

	@Override
	public List<String> weightKeys(Layer layer) {
		final ArrayList<String> keys = new ArrayList<>(WEIGHT_KEYS);

		if (hasLayerNorm(layer)) {
			keys.add(GAIN_KEY);
		}

		return keys;
	}

	@Override
	public List<String> biasKeys(Layer layer) {
		return BIAS_KEYS;
	}

	@Override
	public boolean isWeightParam(Layer layer, String key) {
		return WEIGHT_KEY.equals(key) || RECURRENT_WEIGHT_KEY.equals(key) || GAIN_KEY.equals(key);
	}

	@Override
	public boolean isBiasParam(Layer layer, String key) {
		return BIAS_KEY.equals(key);
	}

	@Override
	public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
		CustomSimpleRNNConf c = (CustomSimpleRNNConf) conf.getLayer();
		val nIn = c.getNIn();
		val nOut = c.getNOut();

		Map<String, INDArray> m;

		if (initializeParams) {
			m = getSubsets(paramsView, nIn, nOut, false, hasLayerNorm(c));
			INDArray w = c.getWeightInitFn().init(nIn, nOut, new long[] { nIn, nOut }, 'f', m.get(WEIGHT_KEY));
			m.put(WEIGHT_KEY, w);

			IWeightInit rwInit = c.getWeightInitFn();

			INDArray rw = rwInit.init(nOut, nOut, new long[] { nOut, nOut }, 'f', m.get(RECURRENT_WEIGHT_KEY));
			m.put(RECURRENT_WEIGHT_KEY, rw);

			m.get(BIAS_KEY).assign(c.getBiasInit());

			if (hasLayerNorm(c)) {
				m.get(GAIN_KEY).assign(c.getGainInit());
			}
		} else {
			m = getSubsets(paramsView, nIn, nOut, true, hasLayerNorm(c));
		}

		conf.addVariable(WEIGHT_KEY);
		conf.addVariable(RECURRENT_WEIGHT_KEY);
		conf.addVariable(BIAS_KEY);
		if (hasLayerNorm(c)) {
			conf.addVariable(GAIN_KEY);
		}

		return m;
	}

	@Override
	public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
		CustomSimpleRNNConf c = (CustomSimpleRNNConf) conf.getLayer();
		val nIn = c.getNIn();
		val nOut = c.getNOut();

		return getSubsets(gradientView, nIn, nOut, true, hasLayerNorm(c));
	}

	private static Map<String, INDArray> getSubsets(INDArray in, long nIn, long nOut, boolean reshape,
			boolean hasLayerNorm) {
		long pos = nIn * nOut;
		INDArray w = in.get(interval(0, 0, true), interval(0, pos));
		INDArray rw = in.get(interval(0, 0, true), interval(pos, pos + nOut * nOut));
		pos += nOut * nOut;
		INDArray b = in.get(interval(0, 0, true), interval(pos, pos + nOut));

		if (reshape) {
			w = w.reshape('f', nIn, nOut);
			rw = rw.reshape('f', nOut, nOut);
		}

		Map<String, INDArray> m = new LinkedHashMap<>();
		m.put(WEIGHT_KEY, w);
		m.put(RECURRENT_WEIGHT_KEY, rw);
		m.put(BIAS_KEY, b);
		if (hasLayerNorm) {
			pos += nOut;
			INDArray g = in.get(interval(0, 0, true), interval(pos, pos + 2 * nOut));
			m.put(GAIN_KEY, g);
		}
		return m;
	}

	protected boolean hasLayerNorm(Layer layer) {
		if (layer instanceof CustomSimpleRNNConf) {
			return ((CustomSimpleRNNConf) layer).hasLayerNorm();
		}
		return false;
	}
}
