
public class Layer {
	
	public Neuron[] neurons; 
	public double[] inputs;
	public double[] weights;
	public double[] outputs;
	public double forward;
	public double testing;
	public Layer(int neuronNumber, int inputsNumber, ActivationFunction activationFunction,double forward,double testing){
		this.neurons=new Neuron[neuronNumber];
		this.inputs=new double[inputsNumber];
		this.weights=new double[inputsNumber];
		this.outputs=new double[neuronNumber];
		this.forward=forward;
		this.testing=testing;
		//////////////////////////////////////////////////////////
		for(int i=0;i<neuronNumber;i++){
			for(int j=0;j<inputsNumber;j++)
				this.weights[j]=10*Math.random()-5;
			this.neurons[i]=new Neuron(this.inputs,10*Math.random()-5, this.weights, activationFunction,this.forward,this.testing);///////////////////////
		}
	}
	
	public void setInputs(double[] inputs){
		for(int i=0;i<neurons.length;i++){
			this.neurons[i].setInputs(inputs);
		}
	}
	
	public void forwardPropagation(){
		for(int i=0;i<this.outputs.length;i++){
			this.outputs[i]=this.neurons[i].forwardPropagation();
			//System.out.println("Out"+this.outputs[i]);
		}
	}
	
	public void testing(){
		for(int i=0;i<this.outputs.length;i++){
			this.outputs[i]=this.neurons[i].testing();
			//System.out.println("Out"+this.outputs[i]);
		}
	}
	
	public void nextLayerPropagation(Layer nextLayer){
		for(int i=0;i<outputs.length;i++){
			nextLayer.setInputs(this.outputs);
		}
	}

}
