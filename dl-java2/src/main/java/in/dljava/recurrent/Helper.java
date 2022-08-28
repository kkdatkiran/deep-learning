package in.dljava.recurrent;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.AbstractLSTM;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.FwdPassReturn;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelper;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.same.TimesOneMinus;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import lombok.val;

@SuppressWarnings("deprecation")
public class Helper {

	static public FwdPassReturn activateHelper(CustomLSTM layer, final NeuralNetConfiguration conf,
			final IActivation gateActivationFn, INDArray input, final INDArray recurrentWeights,
			final INDArray originalInputWeights, final INDArray biases, final boolean training,
			final INDArray originalPrevOutputActivations, final INDArray originalPrevMemCellState, boolean forBackprop,
			boolean forwards, final String inputWeightKey, INDArray maskArray, final boolean hasPeepholeConnections,
			final LSTMHelper helper, final CacheMode cacheMode, final LayerWorkspaceMgr workspaceMgr,
			boolean isHelperAllowFallback) {

		if (input == null || input.length() == 0)
			throw new IllegalArgumentException("Invalid input: not set or 0 length");

		INDArray inputWeights = originalInputWeights;
		INDArray prevOutputActivations = originalPrevOutputActivations;

		if (maskArray != null) {
			maskArray = maskArray.castTo(recurrentWeights.dataType());
		}

		boolean is2dInput = input.rank() < 3;

		input = input.castTo(inputWeights.dataType());

		if ((!is2dInput && (input.size(2) > Integer.MAX_VALUE)) || recurrentWeights.size(0) > Integer.MAX_VALUE
				|| input.size(0) > Integer.MAX_VALUE)
			throw new ND4JArraySizeException();
		int timeSeriesLength = (int) (is2dInput ? 1 : input.size(2));
		int hiddenLayerSize = (int) recurrentWeights.size(0);
		int miniBatchSize = (int) input.size(0);

		INDArray prevMemCellState;
		if (originalPrevMemCellState == null) {
			prevMemCellState = Nd4j.create(inputWeights.dataType(), new long[] { miniBatchSize, hiddenLayerSize }, 'f');
		} else {
			prevMemCellState = originalPrevMemCellState.dup('f');
		}

		INDArray recurrentWeightsIFOG = recurrentWeights.get(all(), interval(0, 4 * hiddenLayerSize)).dup('f');

		INDArray wFFTranspose = null;
		INDArray wOOTranspose = null;
		INDArray wGGTranspose = null;

		if (hasPeepholeConnections) {
			wFFTranspose = recurrentWeights.get(all(), interval(4 * hiddenLayerSize, 4 * hiddenLayerSize + 1))
					.reshape(1, recurrentWeights.size(0));
			wOOTranspose = recurrentWeights.get(all(), interval(4 * hiddenLayerSize + 1, 4 * hiddenLayerSize + 2))
					.reshape(1, recurrentWeights.size(0));
			wGGTranspose = recurrentWeights.get(all(), interval(4 * hiddenLayerSize + 2, 4 * hiddenLayerSize + 3))
					.reshape(1, recurrentWeights.size(0));

			if (timeSeriesLength > 1 || forBackprop) {
				wFFTranspose = Shape.toMmulCompatible(wFFTranspose);
				wOOTranspose = Shape.toMmulCompatible(wOOTranspose);
				wGGTranspose = Shape.toMmulCompatible(wGGTranspose);
			}
		}

		boolean sigmoidGates = gateActivationFn instanceof ActivationSigmoid;
		IActivation afn = layer.layerConf().getActivationFn();
		INDArray outputActivations = null;

		FwdPassReturn toReturn = new FwdPassReturn();
		if (forBackprop) {
			toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];
			toReturn.memCellState = new INDArray[timeSeriesLength];
			toReturn.memCellActivations = new INDArray[timeSeriesLength];
			toReturn.iz = new INDArray[timeSeriesLength];
			toReturn.ia = new INDArray[timeSeriesLength];
			toReturn.fa = new INDArray[timeSeriesLength];
			toReturn.oa = new INDArray[timeSeriesLength];
			toReturn.ga = new INDArray[timeSeriesLength];
			if (!sigmoidGates) {
				toReturn.fz = new INDArray[timeSeriesLength];
				toReturn.oz = new INDArray[timeSeriesLength];
				toReturn.gz = new INDArray[timeSeriesLength];
			}

			if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE)
					&& workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
				try (MemoryWorkspace wsB = workspaceMgr.notifyScopeBorrowed(ArrayType.FF_CACHE)) {
					outputActivations = Nd4j.create(inputWeights.dataType(),
							new long[] { miniBatchSize, hiddenLayerSize, timeSeriesLength }, 'f');
					toReturn.fwdPassOutput = outputActivations;
				}
			} else {
				outputActivations = workspaceMgr.create(ArrayType.ACTIVATIONS, input.dataType(),
						new long[] { miniBatchSize, hiddenLayerSize, timeSeriesLength }, 'f');
				toReturn.fwdPassOutput = outputActivations;
			}
		} else {
			outputActivations = workspaceMgr.create(ArrayType.ACTIVATIONS, input.dataType(),
					new long[] { miniBatchSize, hiddenLayerSize, timeSeriesLength }, 'f');
			toReturn.fwdPassOutput = outputActivations;
		}

		if (input.size(1) != inputWeights.size(0)) {
			throw new DL4JInvalidInputException("Received input with size(1) = " + input.size(1)
					+ " (input array shape = " + Arrays.toString(input.shape())
					+ "); input.size(1) must match layer nIn size (nIn = " + inputWeights.size(0) + ")");
		}
		Preconditions.checkState(prevOutputActivations == null || prevOutputActivations.size(0) == input.size(0),
				"Invalid RNN previous state (last time step activations/initialization): rnnTimeStep with different minibatch size, or forgot to call rnnClearPreviousState between batches?"
						+ " Previous step output = [batch, nIn] = %ndShape, current input = [batch, nIn, seqLength] = %ndShape",
				prevOutputActivations, input);

		if (prevOutputActivations == null) {
			prevOutputActivations = Nd4j.zeros(input.dataType(), new long[] { miniBatchSize, hiddenLayerSize });
		}

		if (helper != null && (layer.getHelperCountFail() == 0 || !isHelperAllowFallback)) {
			FwdPassReturn ret = null;
			try {
				ret = helper.activate(layer, conf, gateActivationFn, input, recurrentWeights, inputWeights, biases,
						training, prevOutputActivations, prevMemCellState, forBackprop, forwards, inputWeightKey,
						maskArray, hasPeepholeConnections, workspaceMgr);
			} catch (ND4JOpProfilerException e) {
				throw e;
			} catch (Exception e) {
				if (e.getMessage().contains("Failed to allocate")) {

					throw e;
				}

				if (isHelperAllowFallback) {
					layer.setHelperCountFail(layer.getHelperCountFail() + 1);

				} else {
					throw new RuntimeException(
							"Error during LSTM MKL/CuDNN helper forward pass - helperAllowFallback() is set to false",
							e);
				}
			}

			if (ret != null) {
				return ret;
			}
		}

		for (int iTimeIndex = 0; iTimeIndex < timeSeriesLength; iTimeIndex++) {
			try (MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.RNN_FF_LOOP_WORKING_MEM)) {
				int time = iTimeIndex;

				if (!forwards) {
					time = timeSeriesLength - iTimeIndex - 1;
				}

				INDArray miniBatchData = (is2dInput ? input : input.tensorAlongDimension(time, 1, 0));
				miniBatchData = Shape.toMmulCompatible(miniBatchData);

				cacheEnter(training, cacheMode, workspaceMgr);

				INDArray ifogActivations = miniBatchData.mmul(inputWeights);
				cacheExit(training, cacheMode, workspaceMgr);

				Nd4j.gemm(prevOutputActivations, recurrentWeightsIFOG, ifogActivations, false, false, 1.0, 1.0);
				ifogActivations.addiRowVector(biases);

				INDArray inputActivations = ifogActivations.get(all(), interval(0, hiddenLayerSize));
				if (forBackprop) {
					if (shouldCache(training, cacheMode, workspaceMgr)) {
						cacheEnter(training, cacheMode, workspaceMgr);
						toReturn.iz[time] = inputActivations.dup('f');
						cacheExit(training, cacheMode, workspaceMgr);
					} else {
						toReturn.iz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, inputActivations, 'f');
					}
				}
				layer.layerConf().getActivationFn().getActivation(inputActivations, training);
				if (forBackprop) {
					if (shouldCache(training, cacheMode, workspaceMgr)) {
						cacheEnter(training, cacheMode, workspaceMgr);
						toReturn.ia[time] = inputActivations.dup('f');
						cacheExit(training, cacheMode, workspaceMgr);
					} else {
						toReturn.ia[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, inputActivations);
					}
				}

				INDArray forgetGateActivations = ifogActivations.get(all(),
						interval(hiddenLayerSize, 2 * hiddenLayerSize));
				if (hasPeepholeConnections) {
					INDArray pmcellWFF = prevMemCellState.dup('f').muliRowVector(wFFTranspose);
					forgetGateActivations.addi(pmcellWFF);
				}

				if (forBackprop && !sigmoidGates) {
					if (shouldCache(training, cacheMode, workspaceMgr)) {
						cacheEnter(training, cacheMode, workspaceMgr);
						toReturn.fz[time] = forgetGateActivations.dup('f');
						cacheExit(training, cacheMode, workspaceMgr);
					} else {
						toReturn.fz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, forgetGateActivations, 'f');
					}
				}
				gateActivationFn.getActivation(forgetGateActivations, training);

				if (forBackprop) {
					if (shouldCache(training, cacheMode, workspaceMgr)) {
						cacheEnter(training, cacheMode, workspaceMgr);
						toReturn.fa[time] = forgetGateActivations.dup('f');
						cacheExit(training, cacheMode, workspaceMgr);
					} else {
						toReturn.fa[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, forgetGateActivations);
					}
				}

				INDArray inputModGateActivations = ifogActivations.get(all(),
						interval(3 * hiddenLayerSize, 4 * hiddenLayerSize));
				if (hasPeepholeConnections) {
					INDArray pmcellWGG = prevMemCellState.dup('f').muliRowVector(wGGTranspose);
					inputModGateActivations.addi(pmcellWGG);
				}
				if (forBackprop && !sigmoidGates) {
					cacheEnter(training, cacheMode, workspaceMgr);
					toReturn.gz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, inputModGateActivations, 'f');
					cacheExit(training, cacheMode, workspaceMgr);
				}
				gateActivationFn.getActivation(inputModGateActivations, training);
				if (forBackprop) {
					if (shouldCache(training, cacheMode, workspaceMgr)) {
						cacheEnter(training, cacheMode, workspaceMgr);
						toReturn.ga[time] = inputModGateActivations.dup('f');
						cacheExit(training, cacheMode, workspaceMgr);
					} else {
						toReturn.ga[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, inputModGateActivations);
					}
				}

				INDArray currentMemoryCellState;
				INDArray inputModMulInput;
				if (forBackprop) {
					cacheEnter(training, cacheMode, workspaceMgr);
					currentMemoryCellState = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, prevMemCellState, 'f')
							.muli(forgetGateActivations);
					cacheExit(training, cacheMode, workspaceMgr);

					inputModMulInput = inputModGateActivations.dup('f').muli(inputActivations);
				} else {
					currentMemoryCellState = workspaceMgr.leverageTo(ArrayType.FF_WORKING_MEM,
							forgetGateActivations.muli(prevMemCellState));
					inputModMulInput = inputModGateActivations.muli(inputActivations);
				}
				currentMemoryCellState.addi(inputModMulInput);

				INDArray outputGateActivations = ifogActivations.get(all(),
						interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));
				if (hasPeepholeConnections) {
					INDArray pmcellWOO = currentMemoryCellState.dup('f').muliRowVector(wOOTranspose);
					outputGateActivations.addi(pmcellWOO);
				}
				if (forBackprop && !sigmoidGates) {
					cacheEnter(training, cacheMode, workspaceMgr);
					toReturn.oz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, outputGateActivations, 'f');
					cacheExit(training, cacheMode, workspaceMgr);
				}
				gateActivationFn.getActivation(outputGateActivations, training);
				if (forBackprop) {
					if (shouldCache(training, cacheMode, workspaceMgr)) {
						cacheEnter(training, cacheMode, workspaceMgr);
						toReturn.oa[time] = outputGateActivations.dup('f');
						cacheExit(training, cacheMode, workspaceMgr);
					} else {
						toReturn.oa[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, outputGateActivations);
					}
				}

				cacheEnter(training, cacheMode, workspaceMgr);

				INDArray currMemoryCellActivation;
				currMemoryCellActivation = workspaceMgr.dup(ArrayType.FF_WORKING_MEM, currentMemoryCellState, 'f');
				currMemoryCellActivation = afn.getActivation(currMemoryCellActivation, training);
				cacheExit(training, cacheMode, workspaceMgr);

				INDArray currHiddenUnitActivations;
				if (forBackprop) {
					cacheEnter(training, cacheMode, workspaceMgr);
					currHiddenUnitActivations = workspaceMgr
							.dup(ArrayType.BP_WORKING_MEM, currMemoryCellActivation, 'f').muli(outputGateActivations);
					cacheExit(training, cacheMode, workspaceMgr);
				} else {
					currHiddenUnitActivations = currMemoryCellActivation.muli(outputGateActivations);
				}

				if (maskArray != null) {

					INDArray timeStepMaskColumn = maskArray.getColumn(time, true);
					currHiddenUnitActivations.muliColumnVector(timeStepMaskColumn);
					currentMemoryCellState.muliColumnVector(timeStepMaskColumn);
				}

				currentMemoryCellState = workspaceMgr.leverageTo(ArrayType.FF_WORKING_MEM, currentMemoryCellState);
				if (forBackprop) {
					toReturn.fwdPassOutputAsArrays[time] = currHiddenUnitActivations;
					toReturn.memCellState[time] = currentMemoryCellState;
					toReturn.memCellActivations[time] = currMemoryCellActivation;

					if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE)
							&& workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
						toReturn.memCellActivations[time] = workspaceMgr.leverageTo(ArrayType.FF_CACHE,
								toReturn.memCellActivations[time]);
						toReturn.memCellState[time] = workspaceMgr.leverageTo(ArrayType.FF_CACHE,
								toReturn.memCellState[time]);
					}

					if (cacheMode != CacheMode.NONE) {
						outputActivations.tensorAlongDimension(time, 1, 0).assign(currHiddenUnitActivations);
					}
				} else {
					outputActivations.tensorAlongDimension(time, 1, 0).assign(currHiddenUnitActivations);
				}

				prevOutputActivations = currHiddenUnitActivations;
				prevMemCellState = currentMemoryCellState;

				toReturn.lastAct = currHiddenUnitActivations;

				toReturn.lastMemCell = currentMemoryCellState;
			}
		}

		toReturn.prevAct = originalPrevOutputActivations;
		toReturn.prevMemCell = originalPrevMemCellState;

		return toReturn;
	}

	private static boolean shouldCache(boolean training, CacheMode cacheMode, LayerWorkspaceMgr workspaceMgr) {
		return training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE)
				&& workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE);
	}

	private static void cacheEnter(boolean training, CacheMode cacheMode, LayerWorkspaceMgr workspaceMgr) {
		if (shouldCache(training, cacheMode, workspaceMgr)) {
			workspaceMgr.notifyScopeBorrowed(ArrayType.FF_CACHE);
		}
	}

	private static void cacheExit(boolean training, CacheMode cacheMode, LayerWorkspaceMgr workspaceMgr) {
		if (shouldCache(training, cacheMode, workspaceMgr)) {
			Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceMgr.getWorkspaceName(ArrayType.FF_CACHE))
					.notifyScopeLeft();
		}
	}

	static public Pair<Gradient, INDArray> backpropGradientHelper(final CustomLSTM layer,
			final NeuralNetConfiguration conf, final IActivation gateActivationFn, INDArray input,
			final INDArray recurrentWeights, final INDArray inputWeights, final INDArray epsilon,
			final boolean truncatedBPTT, final int tbpttBackwardLength, final FwdPassReturn fwdPass,
			final boolean forwards, final String inputWeightKey, final String recurrentWeightKey,
			final String biasWeightKey, final Map<String, INDArray> gradientViews, INDArray maskArray,
			final boolean hasPeepholeConnections, final LSTMHelper helper, final LayerWorkspaceMgr workspaceMgr,
			final boolean isHelperAllowFallback) {

		input = input.castTo(inputWeights.dataType());

		val hiddenLayerSize = recurrentWeights.size(0);
		val prevLayerSize = inputWeights.size(0);
		val miniBatchSize = epsilon.size(0);
		boolean is2dInput = epsilon.rank() < 3;
		val timeSeriesLength = (is2dInput ? 1 : epsilon.size(2));
		INDArray wFFTranspose = null;
		INDArray wOOTranspose = null;
		INDArray wGGTranspose = null;
		if (hasPeepholeConnections) {
			wFFTranspose = recurrentWeights.get(all(), point(4 * hiddenLayerSize)).reshape(1, recurrentWeights.size(0));
			wOOTranspose = recurrentWeights.get(all(), point(4 * hiddenLayerSize + 1)).reshape(1,
					recurrentWeights.size(0));
			wGGTranspose = recurrentWeights.get(all(), point(4 * hiddenLayerSize + 2)).reshape(1,
					recurrentWeights.size(0));
		}

		INDArray wIFOG = recurrentWeights.get(all(), interval(0, 4 * hiddenLayerSize));

		INDArray epsilonNext = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, input.dataType(),
				new long[] { miniBatchSize, prevLayerSize, timeSeriesLength }, 'f');

		INDArray nablaCellStateNext = null;

		INDArray deltaifogNext = Nd4j.create(inputWeights.dataType(), new long[] { miniBatchSize, 4 * hiddenLayerSize },
				'f');
		INDArray deltaiNext = deltaifogNext.get(all(), interval(0, hiddenLayerSize));
		INDArray deltafNext = deltaifogNext.get(all(), interval(hiddenLayerSize, 2 * hiddenLayerSize));
		INDArray deltaoNext = deltaifogNext.get(all(), interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));
		INDArray deltagNext = deltaifogNext.get(all(), interval(3 * hiddenLayerSize, 4 * hiddenLayerSize));

		long endIdx = 0;

		if (truncatedBPTT) {
			endIdx = Math.max(0, timeSeriesLength - tbpttBackwardLength);
		}

		INDArray iwGradientsOut = gradientViews.get(inputWeightKey);
		INDArray rwGradientsOut = gradientViews.get(recurrentWeightKey);
		INDArray bGradientsOut = gradientViews.get(biasWeightKey);
		iwGradientsOut.assign(0);
		rwGradientsOut.assign(0);
		bGradientsOut.assign(0);

		INDArray rwGradientsIFOG = rwGradientsOut.get(all(), interval(0, 4 * hiddenLayerSize));
		INDArray rwGradientsFF = null;
		INDArray rwGradientsOO = null;
		INDArray rwGradientsGG = null;
		if (hasPeepholeConnections) {
			rwGradientsFF = rwGradientsOut.get(all(), NDArrayIndex.point(4 * hiddenLayerSize)).reshape(1,
					recurrentWeights.size(0));
			rwGradientsOO = rwGradientsOut.get(all(), NDArrayIndex.point(4 * hiddenLayerSize + 1)).reshape(1,
					recurrentWeights.size(0));
			rwGradientsGG = rwGradientsOut.get(all(), NDArrayIndex.point(4 * hiddenLayerSize + 2)).reshape(1,
					recurrentWeights.size(0));
		}

		if (helper != null && (layer.getHelperCountFail() == 0 || !isHelperAllowFallback)) {
			Pair<Gradient, INDArray> ret = null;
			try {
				ret = helper.backpropGradient(conf, gateActivationFn, input, recurrentWeights, inputWeights, epsilon,
						truncatedBPTT, tbpttBackwardLength, fwdPass, forwards, inputWeightKey, recurrentWeightKey,
						biasWeightKey, gradientViews, maskArray, hasPeepholeConnections, workspaceMgr);
			} catch (ND4JOpProfilerException e) {
				throw e;
			} catch (Exception e) {
				if (e.getMessage().contains("Failed to allocate")) {

					throw e;
				}

				if (isHelperAllowFallback) {
					layer.setHelperCountFail(layer.getHelperCountFail() + 1);

				} else {
					throw new RuntimeException(
							"Error during LSTM MKL/CuDNN helper backprop - helperAllowFallback() is set to false", e);
				}
			}

			if (ret != null) {
				return ret;
			}
		}

		boolean sigmoidGates = gateActivationFn instanceof ActivationSigmoid;
		IActivation afn = ((org.deeplearning4j.nn.conf.layers.BaseLayer) conf.getLayer()).getActivationFn();

		INDArray timeStepMaskColumn = null;
		for (long iTimeIndex = timeSeriesLength - 1; iTimeIndex >= endIdx; iTimeIndex--) {
			try (MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.RNN_BP_LOOP_WORKING_MEM)) {

				if (iTimeIndex > Integer.MAX_VALUE)
					throw new ND4JArraySizeException();
				int time = (int) iTimeIndex;
				int inext = 1;

				if (!forwards) {
					time = (int) (timeSeriesLength - iTimeIndex - 1);
					inext = -1;
				}

				INDArray nablaCellState;
				if (iTimeIndex != timeSeriesLength - 1 && hasPeepholeConnections) {
					nablaCellState = deltafNext.dup('f').muliRowVector(wFFTranspose);
					nablaCellState.addi(deltagNext.dup('f').muliRowVector(wGGTranspose));
				} else {
					nablaCellState = Nd4j.create(inputWeights.dataType(), new long[] { miniBatchSize, hiddenLayerSize },
							'f');
				}

				INDArray prevMemCellState = (iTimeIndex == 0 ? fwdPass.prevMemCell
						: fwdPass.memCellState[(time - inext)]);
				INDArray prevHiddenUnitActivation = (iTimeIndex == 0 ? fwdPass.prevAct
						: fwdPass.fwdPassOutputAsArrays[(time - inext)]);
				INDArray currMemCellState = fwdPass.memCellState[time];

				INDArray epsilonSlice = (is2dInput ? epsilon : epsilon.tensorAlongDimension(time, 1, 0));
				INDArray nablaOut = Shape.toOffsetZeroCopy(epsilonSlice, 'f');
				if (iTimeIndex != timeSeriesLength - 1) {

					Nd4j.gemm(deltaifogNext, wIFOG, nablaOut, false, true, 1.0, 1.0);
				}

				INDArray sigmahOfS = fwdPass.memCellActivations[time];
				INDArray ao = fwdPass.oa[time];

				INDArray deltao = deltaoNext;
				Nd4j.getExecutioner().exec(new MulOp(nablaOut, sigmahOfS, deltao));
				if (sigmoidGates) {
					INDArray sigmaoPrimeOfZo = Nd4j.getExecutioner().exec(new TimesOneMinus(ao.dup('f')));
					deltao.muli(sigmaoPrimeOfZo);
				} else {
					deltao.assign(gateActivationFn.backprop(fwdPass.oz[time], deltao).getFirst());

				}

				INDArray temp = afn.backprop(currMemCellState.dup('f'), ao.muli(nablaOut)).getFirst();
				nablaCellState.addi(temp);
				if (hasPeepholeConnections) {
					INDArray deltaMulRowWOO = deltao.dup('f').muliRowVector(wOOTranspose);
					nablaCellState.addi(deltaMulRowWOO);
				}
				if (iTimeIndex != timeSeriesLength - 1) {
					INDArray nextForgetGateAs = fwdPass.fa[time + inext];
					nablaCellState.addi(nextForgetGateAs.muli(nablaCellStateNext));
				}

				nablaCellStateNext = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, nablaCellState);

				INDArray af = fwdPass.fa[time];
				INDArray deltaf = null;
				if (iTimeIndex > 0 || prevMemCellState != null) {

					deltaf = deltafNext;
					if (sigmoidGates) {
						Nd4j.getExecutioner().exec(new TimesOneMinus(af, deltaf));
						deltaf.muli(nablaCellState);
						deltaf.muli(prevMemCellState);
					} else {
						INDArray temp2 = nablaCellState.mul(prevMemCellState);
						deltaf.assign(gateActivationFn.backprop(fwdPass.fz[time].dup('f'), temp2).getFirst());

					}
				}

				INDArray ag = fwdPass.ga[time];
				INDArray ai = fwdPass.ia[time];
				INDArray deltag = deltagNext;
				if (sigmoidGates) {
					Nd4j.getExecutioner().exec(new TimesOneMinus(ag, deltag));
					deltag.muli(ai);
					deltag.muli(nablaCellState);
				} else {
					INDArray temp2 = Nd4j.getExecutioner().exec(new MulOp(ai, nablaCellState,
							Nd4j.createUninitialized(inputWeights.dataType(), ai.shape(), 'f')))[0];
					deltag.assign(gateActivationFn.backprop(fwdPass.gz[time], temp2).getFirst());

				}

				INDArray zi = fwdPass.iz[time];
				INDArray deltai = deltaiNext;
				temp = Nd4j.getExecutioner().exec(new MulOp(ag, nablaCellState,
						Nd4j.createUninitialized(inputWeights.dataType(), deltai.shape(), 'f')))[0];
				deltai.assign(afn.backprop(zi, temp).getFirst());

				if (maskArray != null) {

					timeStepMaskColumn = maskArray.getColumn(time, true);
					deltaifogNext.muli(timeStepMaskColumn);

				}

				INDArray prevLayerActivationSlice = Shape
						.toMmulCompatible(is2dInput ? input : input.tensorAlongDimension(time, 1, 0));
				if (iTimeIndex > 0 || prevHiddenUnitActivation != null) {

					Nd4j.gemm(prevLayerActivationSlice, deltaifogNext, iwGradientsOut, true, false, 1.0, 1.0);
				} else {
					INDArray iwGradients_i = iwGradientsOut.get(all(), interval(0, hiddenLayerSize));
					Nd4j.gemm(prevLayerActivationSlice, deltai, iwGradients_i, true, false, 1.0, 1.0);
					INDArray iwGradients_og = iwGradientsOut.get(all(),
							interval(2 * hiddenLayerSize, 4 * hiddenLayerSize));
					INDArray deltaog = deltaifogNext.get(all(), interval(2 * hiddenLayerSize, 4 * hiddenLayerSize));
					Nd4j.gemm(prevLayerActivationSlice, deltaog, iwGradients_og, true, false, 1.0, 1.0);
				}

				if (iTimeIndex > 0 || prevHiddenUnitActivation != null) {

					Nd4j.gemm(prevHiddenUnitActivation, deltaifogNext, rwGradientsIFOG, true, false, 1.0, 1.0);

					if (hasPeepholeConnections) {
						INDArray dLdwFF = deltaf.dup('f').muli(prevMemCellState).sum(true, 0);
						rwGradientsFF.addi(dLdwFF);
						INDArray dLdwGG = deltag.dup('f').muli(prevMemCellState).sum(true, 0);
						rwGradientsGG.addi(dLdwGG);
					}
				}

				if (hasPeepholeConnections) {
					INDArray dLdwOO = deltao.dup('f').muli(currMemCellState).sum(true, 0);
					rwGradientsOO.addi(dLdwOO);
				}

				if (iTimeIndex > 0 || prevHiddenUnitActivation != null) {

					bGradientsOut.addi(deltaifogNext.sum(true, 0));
				} else {
					bGradientsOut.get(interval(0, 0, true), interval(0, hiddenLayerSize)).addi(deltai.sum(true, 0));
					INDArray ogBiasToAdd = deltaifogNext.get(all(), interval(2 * hiddenLayerSize, 4 * hiddenLayerSize))
							.sum(true, 0);
					INDArray ogBiasGrad = bGradientsOut.get(interval(0, 0, true),
							interval(2 * hiddenLayerSize, 4 * hiddenLayerSize));
					ogBiasGrad.addi(ogBiasToAdd);
				}

				INDArray epsilonNextSlice = epsilonNext.tensorAlongDimension(time, 1, 0);
				if (iTimeIndex > 0 || prevHiddenUnitActivation != null) {

					Nd4j.gemm(deltaifogNext, inputWeights, epsilonNextSlice, false, true, 1.0, 1.0);
				} else {

					INDArray wi = inputWeights.get(all(), interval(0, hiddenLayerSize));
					Nd4j.gemm(deltai, wi, epsilonNextSlice, false, true, 1.0, 1.0);
					INDArray deltaog = deltaifogNext.get(all(), interval(2 * hiddenLayerSize, 4 * hiddenLayerSize));
					INDArray wog = inputWeights.get(all(), interval(2 * hiddenLayerSize, 4 * hiddenLayerSize));
					Nd4j.gemm(deltaog, wog, epsilonNextSlice, false, true, 1.0, 1.0);
				}

				if (maskArray != null) {

					epsilonNextSlice.muli(timeStepMaskColumn);
				}
			}
		}

		Gradient retGradient = new DefaultGradient();
		retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut);
		retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
		retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut);

		return new Pair<>(retGradient, epsilonNext);
	}

	public static LayerMemoryReport getMemoryReport(AbstractLSTM lstmLayer, InputType inputType) {
		boolean isGraves = lstmLayer instanceof org.deeplearning4j.nn.conf.layers.GravesLSTM;
		return getMemoryReport(isGraves, lstmLayer, inputType);
	}

	public static LayerMemoryReport getMemoryReport(GravesBidirectionalLSTM lstmLayer, InputType inputType) {
		LayerMemoryReport r = getMemoryReport(true, lstmLayer, inputType);

		Map<CacheMode, Long> fixedTrain = new HashMap<>();
		Map<CacheMode, Long> varTrain = new HashMap<>();
		Map<CacheMode, Long> cacheFixed = new HashMap<>();
		Map<CacheMode, Long> cacheVar = new HashMap<>();
		for (CacheMode cm : CacheMode.values()) {
			fixedTrain.put(cm, 2 * r.getWorkingMemoryFixedTrain().get(cm));
			varTrain.put(cm, 2 * r.getWorkingMemoryVariableTrain().get(cm));
			cacheFixed.put(cm, 2 * r.getCacheModeMemFixed().get(cm));
			cacheVar.put(cm, 2 * r.getCacheModeMemVariablePerEx().get(cm));
		}

		return new LayerMemoryReport.Builder(r.getLayerName(), r.getClass(), r.getInputType(), r.getOutputType())
				.standardMemory(2 * r.getParameterSize(), 2 * r.getUpdaterStateSize())
				.workingMemory(2 * r.getWorkingMemoryFixedInference(), 2 * r.getWorkingMemoryVariableInference(),
						fixedTrain, varTrain)
				.cacheMemory(cacheFixed, cacheVar).build();
	}

	public static LayerMemoryReport getMemoryReport(boolean isGraves,
			org.deeplearning4j.nn.conf.layers.FeedForwardLayer lstmLayer, InputType inputType) {

		InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;
		val tsLength = itr.getTimeSeriesLength();

		InputType outputType = lstmLayer.getOutputType(-1, inputType);

		val numParams = lstmLayer.initializer().numParams(lstmLayer);
		int updaterSize = (int) lstmLayer.getIUpdater().stateSize(numParams);

		val workingMemInferencePerEx = tsLength * 4 * lstmLayer.getNOut();

		val fwdPassPerTimeStepTrainCache = tsLength * 6 * lstmLayer.getNOut();

		val backpropWorkingSpace = (isGraves ? 9 : 6) * tsLength * lstmLayer.getNOut();

		Map<CacheMode, Long> trainVariable = new HashMap<>();
		Map<CacheMode, Long> cacheVariable = new HashMap<>();
		for (CacheMode cm : CacheMode.values()) {
			long trainWorking;
			long cacheMem;

			if (cm == CacheMode.NONE) {
				trainWorking = workingMemInferencePerEx + fwdPassPerTimeStepTrainCache + backpropWorkingSpace;
				cacheMem = 0;
			} else {
				trainWorking = workingMemInferencePerEx + backpropWorkingSpace;
				cacheMem = fwdPassPerTimeStepTrainCache;
			}

			trainVariable.put(cm, trainWorking);
			cacheVariable.put(cm, cacheMem);
		}

		return new LayerMemoryReport.Builder(null, lstmLayer.getClass(), inputType, outputType)
				.standardMemory(numParams, updaterSize)
				.workingMemory(0, workingMemInferencePerEx, MemoryReport.CACHE_MODE_ALL_ZEROS, trainVariable)
				.cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, cacheVariable).build();
	}
}
