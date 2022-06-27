package in.dljava.layers;

import in.dljava.functions.initializers.InitializerFunction;
import in.dljava.funtions.activation.ActivationFunction;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.experimental.Accessors;

@Data
@Accessors(chain = true)
@AllArgsConstructor
public class Dense implements Layer {

	private int units;
	private ActivationFunction activation = ActivationFunction.LINEAR;
	private boolean useBias = false;
	private InitializerFunction kernelInitializer = InitializerFunction.ZEROS;
	private InitializerFunction biasInitializer = InitializerFunction.ZEROS;
	
	public Dense(int units) {
		
		this.units = units;
	}
}
