package in.dljava.layer;

import in.dljava.activation.ActivationFunction;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class DenseLayer extends Layer {

	private ActivationFunction activation;

	public DenseLayer(int units) {
		this.units = units;
	}

	public Layer setActivation(ActivationFunction af) {
		this.activation = af;
		return this;
	}
}
