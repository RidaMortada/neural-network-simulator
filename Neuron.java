

/**
 * 
 */

/**
 * @author PC
 *
 */
public class Neuron {
	public double[] inputs ;
	public double[] weights;
	private double delta;
	public double bias;
	public double forward;
	public double testing;
	ActivationFunction sigmoid;
	public Neuron(double[] inputs, double bias, double[] weights, ActivationFunction sigmoid,double forward, double testing){
		this.inputs=new double[inputs.length];
		this.weights=new double[weights.length];
		this.sigmoid=sigmoid;
		this.bias=bias;
		this.forward=forward;
		this.testing=testing;
		for(int i=0;i<inputs.length;i++){
			this.inputs[i]=inputs[i];
			this.weights[i]=weights[i];
		}
	}
	
	public void setInputs(double[] inputs){
		for(int i=0;i<this.inputs.length;i++)
			this.inputs[i]=inputs[i];
	}
	
	public double forwardPropagation(){
		double sum=0;
		for(int i=0;i<inputs.length;i++){
			sum+=this.inputs[i]*this.weights[i];
		}
		sum+=this.bias;
		//System.out.println("sum "+this.sigmoid.getOutput(sum));
		return this.sigmoid.getOutput(sum,this.forward);
	}
	
	public double testing(){
		double sum=0;
		for(int i=0;i<inputs.length;i++){
			sum+=this.inputs[i]*this.weights[i];
		}
		sum+=this.bias;
		//System.out.println("sum "+this.sigmoid.getOutput(sum));
		return this.sigmoid.getOutput(sum,this.testing);
	}

	/**
	 * @return the delta
	 */
	public double getDelta() {
		return delta;
	}

	/**
	 * @param delta the delta to set
	 */
	public void setDelta(double delta) {
		this.delta = delta;
	}
	

}
