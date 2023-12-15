/**
 * 
 */

/**
 * @author PC
 *
 */
public class MainNetwork {

	/**
	 * @param args
	 */
	private int[] neuronsPerLayer;
	private Layer[] layers;
	public double forward=0.1;
	public double testing=10;
	public MainNetwork(int inputsNumber,int outputsNumber,int[] neuronsPerLayer, ActivationFunction activationFunction){
		this.neuronsPerLayer=neuronsPerLayer;
		this.layers=new Layer[this.neuronsPerLayer.length];
		this.layers[0]=new Layer(this.neuronsPerLayer[0], inputsNumber, activationFunction,this.forward,this.testing);
		for(int i=1;i<neuronsPerLayer.length;i++){
			this.layers[i]=new Layer(this.neuronsPerLayer[i], this.neuronsPerLayer[i-1], activationFunction,this.forward,this.testing);
		}
	}
	
	public double[] forwardPropagation(double[]inputs){
		for(int i=0;i<inputs.length;i++){
			this.layers[0].setInputs(inputs);
			//System.out.println("in "+this.layers[0].inputs[i]);
		}
		for(int i=0;i<neuronsPerLayer.length;i++){
			this.layers[i].forwardPropagation();
			if(i+1<neuronsPerLayer.length){
				//System.out.println("OK");
				this.layers[i].nextLayerPropagation(this.layers[i+1]);
			}
		}
		return this.layers[neuronsPerLayer.length-1].outputs;
	}
	
	public double[] testing(double[]inputs){
		for(int i=0;i<inputs.length;i++){
			this.layers[0].setInputs(inputs);
			//System.out.println("in "+this.layers[0].inputs[i]);
		}
		for(int i=0;i<neuronsPerLayer.length-1;i++){
			this.layers[i].testing();
			if(i+1<neuronsPerLayer.length){
				//System.out.println("OK");
				this.layers[i].nextLayerPropagation(this.layers[i+1]);
			}
		}
		this.layers[neuronsPerLayer.length-1].testing();
		
		for(int i=0;i<this.layers[neuronsPerLayer.length-1].outputs.length;i++){
			if(this.layers[neuronsPerLayer.length-1].outputs[i]<0.0001)
				this.layers[neuronsPerLayer.length-1].outputs[i]=0;
		}
		return this.layers[neuronsPerLayer.length-1].outputs;
	}
	
	public void backPropagation(double[]targets){
		double[] layerOutputs=layers[layers.length-1].outputs;
		for(int i=0;i<layerOutputs.length;i++){
			double delta=2*(layerOutputs[i]-targets[i])*layerOutputs[i]*(1-layerOutputs[i]);
			layers[layers.length-1].neurons[i].setDelta(delta);
			//System.out.println("delta3 "+delta);
		}
		
		for(int i=layers.length-2;i>=0;i--){
			for(int j=0;j<layers[i].neurons.length;j++){
				double sum=0;
				for(int k=0;k<layers[i+1].outputs.length;k++){
					sum+=layers[i+1].neurons[k].getDelta()*layers[i+1].neurons[k].weights[j];
				}
				double delta=sum*layers[i].outputs[j]*(1-layers[i].outputs[j]);
				layers[i].neurons[j].setDelta(delta);
				//System.out.println("delta "+delta);
				
			}
		}
	}
	
	public void weightsUpdate(double learningRate){
		for(int i=0;i<this.layers.length;i++){
			for(int j=0;j<this.layers[i].neurons.length;j++){
				this.layers[i].neurons[j].bias-=learningRate*this.layers[i].neurons[j].getDelta();
				//System.out.println("delta "+learningRate*this.layers[i].neurons[j].getDelta());
				for(int k=0;k<this.layers[i].neurons[j].inputs.length;k++){
					//System.out.println("in "+this.layers[i].neurons[j].inputs[k]);
					double costDerivative= this.layers[i].neurons[j].getDelta()*this.layers[i].neurons[j].inputs[k];
					//System.out.println("w "+this.layers[i].neurons[j].weights[k]);
					this.layers[i].neurons[j].weights[k]-=learningRate*costDerivative;
					//System.out.println(learningRate+"*"+this.layers[i].neurons[j].getDelta()+"*"+this.layers[i].neurons[j].inputs[k]);
					//System.out.println("w "+this.layers[i].neurons[j].weights[k]);
				}
			}
		}
	}
	
	public double costFunction(double[]targets){
		double cost=0;
		double[] layerOutputs=layers[layers.length-1].outputs;
		for(int i=0;i<layerOutputs.length;i++){
			if((targets[i]-layerOutputs[i])*(targets[i]-layerOutputs[i])>cost){
				cost=(targets[i]-layerOutputs[i])*(targets[i]-layerOutputs[i]);
			}
		}
		return cost;
	}
	
	public void learning(double[][] inputsMatrix, double[][] targetsColumn, double learningRate, double targetCost){
		double maxCost=0;
		for(int i=0; i< inputsMatrix.length;i++){
			this.forwardPropagation(inputsMatrix[i]);
			if(this.costFunction(targetsColumn[i])>maxCost)
				maxCost=this.costFunction(targetsColumn[i]);
			
		}
		int watchDog=0;
		while(maxCost>targetCost){
			//System.out.println("max cost "+maxCost);
			watchDog++;
			if(watchDog==100000){
				//System.out.println("reset");
				watchDog=0;
				for(int i=0;i<this.layers.length;i++){
					for(int j=0;j<this.layers[i].neurons.length;j++){
						this.layers[i].neurons[j].bias=10*Math.random()-5;
						for(int k=0;k<this.layers[i].neurons[j].weights.length;k++){
							layers[i].neurons[j].weights[k]=10*Math.random()-5;
						}
					}
				}
			}
			maxCost=0;
			for(int i=0; i< inputsMatrix.length;i++){
				this.forwardPropagation(inputsMatrix[i]);
				if(this.costFunction(targetsColumn[i])>maxCost)
					maxCost=this.costFunction(targetsColumn[i]);
				this.backPropagation(targetsColumn[i]);
				//this.backPropagation(targetsColumn[i]);
				this.weightsUpdate(learningRate);
				//this.forwardPropagation(inputsMatrix[i]);
				
				
			}
		}
		
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int[] neuronsPerLayer = {10,4};
		MainNetwork mn=new MainNetwork(2, 4,neuronsPerLayer, new Sigmoid());
		
		/*
		
		double[][] trainingInputs={{0,0},{0,1},{1,0},{1,1}};
		double[][] targetOutputs={{0},{1},{1},{0}};
		
		double[] targ={0};
		*/
	    /*
		double[][] trainingInputs={{1},{2},{3},{4},{5},{6},{7}};
		double[][] targetOutputs={{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
		*/
		double[][] trainingInputs={{0.5,0.5},{1.5,0.5},{2.5,0.5},{0.5,1.5},{1.5,1.5},{2.5,1.5},{0.5,2.5},{1.5,2.5},{2.5,2.5}};
		double[][] targetOutputs={{0,0,0,1},{0,0,1,0},{0,0,1,1},{0,1,0,0},{0,1,0,1},{0,1,1,0},{0,1,1,1},{1,0,0,0},{1,0,0,1}};
		
		mn.learning(trainingInputs, targetOutputs, 0.03, 0.01);
		
		System.out.println(trainingInputs[0][0]+"  "+trainingInputs[0][1]+" = "+mn.forwardPropagation(trainingInputs[0])[0]+" "+mn.forwardPropagation(trainingInputs[0])[1]+" "+mn.forwardPropagation(trainingInputs[0])[2]+" "+mn.forwardPropagation(trainingInputs[0])[3]);
	
		System.out.println(trainingInputs[1][0]+"  "+trainingInputs[1][1]+" = "+mn.forwardPropagation(trainingInputs[1])[0]+" "+mn.forwardPropagation(trainingInputs[1])[1]+" "+mn.forwardPropagation(trainingInputs[1])[2]+" "+mn.forwardPropagation(trainingInputs[1])[3]);
		
		System.out.println(trainingInputs[2][0]+"  "+trainingInputs[2][1]+" = "+mn.forwardPropagation(trainingInputs[2])[0]+" "+mn.forwardPropagation(trainingInputs[2])[1]+" "+mn.forwardPropagation(trainingInputs[2])[2]+" "+mn.forwardPropagation(trainingInputs[2])[3]);
		
		System.out.println(trainingInputs[3][0]+"  "+trainingInputs[3][1]+" = "+mn.forwardPropagation(trainingInputs[3])[0]+" "+mn.forwardPropagation(trainingInputs[3])[1]+" "+mn.forwardPropagation(trainingInputs[3])[2]+" "+mn.forwardPropagation(trainingInputs[3])[3]);
		
		System.out.println(trainingInputs[4][0]+"  "+trainingInputs[4][1]+" = "+mn.forwardPropagation(trainingInputs[4])[0]+" "+mn.forwardPropagation(trainingInputs[4])[1]+" "+mn.forwardPropagation(trainingInputs[4])[2]+" "+mn.forwardPropagation(trainingInputs[4])[3]);
		
		System.out.println(trainingInputs[5][0]+"  "+trainingInputs[5][1]+" = "+mn.forwardPropagation(trainingInputs[5])[0]+" "+mn.forwardPropagation(trainingInputs[5])[1]+" "+mn.forwardPropagation(trainingInputs[5])[2]+" "+mn.forwardPropagation(trainingInputs[5])[3]);
		
		System.out.println(trainingInputs[6][0]+"  "+trainingInputs[6][1]+" = "+mn.forwardPropagation(trainingInputs[6])[0]+" "+mn.forwardPropagation(trainingInputs[6])[1]+" "+mn.forwardPropagation(trainingInputs[6])[2]+" "+mn.forwardPropagation(trainingInputs[6])[3]);
		
		System.out.println(trainingInputs[7][0]+"  "+trainingInputs[7][1]+" = "+mn.forwardPropagation(trainingInputs[7])[0]+" "+mn.forwardPropagation(trainingInputs[7])[1]+" "+mn.forwardPropagation(trainingInputs[7])[2]+" "+mn.forwardPropagation(trainingInputs[7])[3]);
		
		System.out.println(trainingInputs[8][0]+"  "+trainingInputs[8][1]+" = "+mn.forwardPropagation(trainingInputs[8])[0]+" "+mn.forwardPropagation(trainingInputs[8])[1]+" "+mn.forwardPropagation(trainingInputs[8])[2]+" "+mn.forwardPropagation(trainingInputs[8])[3]);
		
		System.out.println();
		
		System.out.println(trainingInputs[0][0]+"  "+trainingInputs[0][1]+" = "+mn.testing(trainingInputs[0])[0]+" "+mn.testing(trainingInputs[0])[1]+" "+mn.testing(trainingInputs[0])[2]+" "+mn.testing(trainingInputs[0])[3]);
		System.out.println(trainingInputs[1][0]+"  "+trainingInputs[1][1]+" = "+mn.testing(trainingInputs[1])[0]+" "+mn.testing(trainingInputs[1])[1]+" "+mn.testing(trainingInputs[1])[2]+" "+mn.testing(trainingInputs[1])[3]);
		System.out.println(trainingInputs[2][0]+"  "+trainingInputs[2][1]+" = "+mn.testing(trainingInputs[2])[0]+" "+mn.testing(trainingInputs[2])[1]+" "+mn.testing(trainingInputs[2])[2]+" "+mn.testing(trainingInputs[2])[3]);
		System.out.println(trainingInputs[3][0]+"  "+trainingInputs[3][1]+" = "+mn.testing(trainingInputs[3])[0]+" "+mn.testing(trainingInputs[3])[1]+" "+mn.testing(trainingInputs[3])[2]+" "+mn.testing(trainingInputs[3])[3]);
		System.out.println(trainingInputs[4][0]+"  "+trainingInputs[4][1]+" = "+mn.testing(trainingInputs[4])[0]+" "+mn.testing(trainingInputs[4])[1]+" "+mn.testing(trainingInputs[4])[2]+" "+mn.testing(trainingInputs[4])[3]);
		System.out.println(trainingInputs[5][0]+"  "+trainingInputs[5][1]+" = "+mn.testing(trainingInputs[5])[0]+" "+mn.testing(trainingInputs[5])[1]+" "+mn.testing(trainingInputs[5])[2]+" "+mn.testing(trainingInputs[5])[3]);
		System.out.println(trainingInputs[6][0]+"  "+trainingInputs[6][1]+" = "+mn.testing(trainingInputs[6])[0]+" "+mn.testing(trainingInputs[6])[1]+" "+mn.testing(trainingInputs[6])[2]+" "+mn.testing(trainingInputs[6])[3]);
		System.out.println(trainingInputs[7][0]+"  "+trainingInputs[7][1]+" = "+mn.testing(trainingInputs[7])[0]+" "+mn.testing(trainingInputs[7])[1]+" "+mn.testing(trainingInputs[7])[2]+" "+mn.testing(trainingInputs[7])[3]);
		System.out.println(trainingInputs[8][0]+"  "+trainingInputs[8][1]+" = "+mn.testing(trainingInputs[8])[0]+" "+mn.testing(trainingInputs[8])[1]+" "+mn.testing(trainingInputs[8])[2]+" "+mn.testing(trainingInputs[8])[3]);
		System.out.println();
		
		
		
		/*
		
		mn.forwardPropagation(trainingInputs[0]);
		System.out.print(mn.layers[1].neurons[0].inputs[0]);
		System.out.print(" "+mn.layers[1].neurons[0].inputs[1]);
		System.out.println(" "+mn.layers[1].neurons[0].forwardPropagation());
		targ[0]=0;
		mn.backPropagation(targ);
		mn.weightsUpdate(1);
		System.out.println(mn.costFunction(targ));
		
		mn.forwardPropagation(trainingInputs[1]);
		System.out.print(mn.layers[1].neurons[0].inputs[0]);
		System.out.print(" "+mn.layers[1].neurons[0].inputs[1]);
		System.out.println(" "+mn.layers[1].neurons[0].forwardPropagation());
		targ[0]=1;
		mn.backPropagation(targ);
		mn.weightsUpdate(1);
		System.out.println(mn.costFunction(targ));
		
		
		mn.forwardPropagation(trainingInputs[2]);
		System.out.print(mn.layers[1].neurons[0].inputs[0]);
		System.out.print(" "+mn.layers[1].neurons[0].inputs[1]);
		System.out.println(" "+mn.layers[1].neurons[0].forwardPropagation());
		targ[0]=1;
		mn.backPropagation(targ);
		mn.weightsUpdate(1);
		System.out.println(mn.costFunction(targ));
		
		
		mn.forwardPropagation(trainingInputs[3]);
		System.out.print(mn.layers[1].neurons[0].inputs[0]);
		System.out.print(" "+mn.layers[1].neurons[0].inputs[1]);
		System.out.println(" "+mn.layers[1].neurons[0].forwardPropagation());
		targ[0]=0;
		mn.backPropagation(targ);
		mn.weightsUpdate(1);
		System.out.println(mn.costFunction(targ));
		
		
		
		
		
		*/
		
		
		
		/*if(neuronsPerLayer.length==2)
			return;*/
	/*	
		System.out.println(trainingInputs[0][0]+" = "+mn.testing(trainingInputs[0])[0]+" "+mn.testing(trainingInputs[0])[1]+" "+mn.testing(trainingInputs[0])[2]);
		System.out.println(trainingInputs[1][0]+" = "+mn.testing(trainingInputs[1])[0]+" "+mn.testing(trainingInputs[1])[1]+" "+mn.testing(trainingInputs[1])[2]);
		System.out.println(trainingInputs[2][0]+" = "+mn.testing(trainingInputs[2])[0]+" "+mn.testing(trainingInputs[2])[1]+" "+mn.testing(trainingInputs[2])[2]);
		System.out.println(trainingInputs[3][0]+" = "+mn.testing(trainingInputs[3])[0]+" "+mn.testing(trainingInputs[3])[1]+" "+mn.testing(trainingInputs[3])[2]);
		System.out.println(trainingInputs[4][0]+" = "+mn.testing(trainingInputs[4])[0]+" "+mn.testing(trainingInputs[4])[1]+" "+mn.testing(trainingInputs[4])[2]);
		System.out.println(trainingInputs[5][0]+" = "+mn.testing(trainingInputs[5])[0]+" "+mn.testing(trainingInputs[5])[1]+" "+mn.testing(trainingInputs[5])[2]);
		System.out.println(trainingInputs[6][0]+" = "+mn.testing(trainingInputs[6])[0]+" "+mn.testing(trainingInputs[6])[1]+" "+mn.testing(trainingInputs[6])[2]);
		
		System.out.println();
				
		mn.learning(trainingInputs, targetOutputs, 0.1, 0.01);
		
		System.out.println(trainingInputs[0][0]+" = "+mn.testing(trainingInputs[0])[0]+" "+mn.testing(trainingInputs[0])[1]+" "+mn.testing(trainingInputs[0])[2]);
		System.out.println(trainingInputs[1][0]+" = "+mn.testing(trainingInputs[1])[0]+" "+mn.testing(trainingInputs[1])[1]+" "+mn.testing(trainingInputs[1])[2]);
		System.out.println(trainingInputs[2][0]+" = "+mn.testing(trainingInputs[2])[0]+" "+mn.testing(trainingInputs[2])[1]+" "+mn.testing(trainingInputs[2])[2]);
		System.out.println(trainingInputs[3][0]+" = "+mn.testing(trainingInputs[3])[0]+" "+mn.testing(trainingInputs[3])[1]+" "+mn.testing(trainingInputs[3])[2]);
		System.out.println(trainingInputs[4][0]+" = "+mn.testing(trainingInputs[4])[0]+" "+mn.testing(trainingInputs[4])[1]+" "+mn.testing(trainingInputs[4])[2]);
		System.out.println(trainingInputs[5][0]+" = "+mn.testing(trainingInputs[5])[0]+" "+mn.testing(trainingInputs[5])[1]+" "+mn.testing(trainingInputs[5])[2]);
		System.out.println(trainingInputs[6][0]+" = "+mn.testing(trainingInputs[6])[0]+" "+mn.testing(trainingInputs[6])[1]+" "+mn.testing(trainingInputs[6])[2]);
		System.out.println();
		
		
		//System.out.println(trainingInputs[0][0]+" = "+mn.testing(trainingInputs[0])[0]+" "+mn.testing(trainingInputs[0])[1]+" "+mn.testing(trainingInputs[0])[2]);
		mn.testing(trainingInputs[0]);
		for(int i=0; i<mn.layers[0].outputs.length;i++){
			if(mn.layers[0].outputs[i]<0.0001){
				System.out.print(0.0+" ");
				continue;
			}
			System.out.print(mn.layers[0].outputs[i]+" ");
		}
		System.out.println();
		//System.out.println(trainingInputs[1][0]+" = "+mn.testing(trainingInputs[1])[0]+" "+mn.testing(trainingInputs[1])[1]+" "+mn.testing(trainingInputs[1])[2]);
		mn.testing(trainingInputs[1]);
		for(int i=0; i<mn.layers[0].outputs.length;i++){
			if(mn.layers[0].outputs[i]<0.0001){
				System.out.print(0.0+" ");
				continue;
			}
			System.out.print(mn.layers[0].outputs[i]+" ");
		}
		System.out.println();
		//System.out.println(trainingInputs[2][0]+" = "+mn.testing(trainingInputs[2])[0]+" "+mn.testing(trainingInputs[2])[1]+" "+mn.testing(trainingInputs[2])[2]);
		mn.testing(trainingInputs[2]);
		for(int i=0; i<mn.layers[0].outputs.length;i++){
			if(mn.layers[0].outputs[i]<0.0001){
				System.out.print(0.0+" ");
				continue;
			}
			System.out.print(mn.layers[0].outputs[i]+" ");
		}
		System.out.println();
		//System.out.println(trainingInputs[3][0]+" = "+mn.testing(trainingInputs[3])[0]+" "+mn.testing(trainingInputs[3])[1]+" "+mn.testing(trainingInputs[3])[2]);
		mn.testing(trainingInputs[3]);
		for(int i=0; i<mn.layers[0].outputs.length;i++){
			if(mn.layers[0].outputs[i]<0.0001){
				System.out.print(0.0+" ");
				continue;
			}
			System.out.print(mn.layers[0].outputs[i]+" ");
		}
		System.out.println();
		//System.out.println(trainingInputs[4][0]+" = "+mn.testing(trainingInputs[4])[0]+" "+mn.testing(trainingInputs[4])[1]+" "+mn.testing(trainingInputs[4])[2]);
		mn.testing(trainingInputs[4]);
		for(int i=0; i<mn.layers[0].outputs.length;i++){
			if(mn.layers[0].outputs[i]<0.0001){
				System.out.print(0.0+" ");
				continue;
			}
			System.out.print(mn.layers[0].outputs[i]+" ");
		}
		System.out.println();
		//System.out.println(trainingInputs[5][0]+" = "+mn.testing(trainingInputs[5])[0]+" "+mn.testing(trainingInputs[5])[1]+" "+mn.testing(trainingInputs[5])[2]);
		mn.testing(trainingInputs[5]);
		for(int i=0; i<mn.layers[0].outputs.length;i++){
			if(mn.layers[0].outputs[i]<0.0001){
				System.out.print(0.0+" ");
				continue;
			}
			System.out.print(mn.layers[0].outputs[i]+" ");
		}
		//System.out.println(trainingInputs[6][0]+" = "+mn.testing(trainingInputs[6])[0]+" "+mn.testing(trainingInputs[6])[1]+" "+mn.testing(trainingInputs[6])[2]);
		System.out.println();
		mn.testing(trainingInputs[6]);
		for(int i=0; i<mn.layers[0].outputs.length;i++){
			if(mn.layers[0].outputs[i]<0.0001){
				System.out.print(0.0+" ");
				continue;
			}
			System.out.print(mn.layers[0].outputs[i]+" ");
		}
		
		
		System.out.println("\n");
		
		
		System.out.println("\n");
		System.out.println("b= "+mn.layers[mn.layers.length-1].neurons[2].bias);
		for(int i=0;i<mn.layers[mn.layers.length-1].neurons[1].weights.length;i++){
			System.out.println("w"+i+"= "+mn.layers[mn.layers.length-1].neurons[2].weights[i]);
		}
		
		System.out.println("\n");
		System.out.println("b= "+mn.layers[mn.layers.length-1].neurons[1].bias);
		for(int i=0;i<mn.layers[mn.layers.length-1].neurons[1].weights.length;i++){
			System.out.println("w"+i+"= "+mn.layers[mn.layers.length-1].neurons[1].weights[i]);
		}
		
		System.out.println("\n");
		System.out.println("b= "+mn.layers[mn.layers.length-1].neurons[0].bias);
		for(int i=0;i<mn.layers[mn.layers.length-1].neurons[1].weights.length;i++){
			System.out.println("w"+i+"= "+mn.layers[mn.layers.length-1].neurons[0].weights[i]);
		}
	
		
		/*
		for(int r=0;r<1;r++){
		
		mn.backPropagation(targetOutputs[0]);
		mn.weightsUpdate(0.1);
		for(int i=0;i<mn.layers.length;i++){
			for(int j=0;j<mn.layers[i].neurons.length;j++){
				for(int k=0;k<mn.layers[i].neurons[j].weights.length;k++){
					System.out.print("weight "+mn.layers[i].neurons[j].weights[k]+" ");
				}
				System.out.println();
				System.out.println("bias "+mn.layers[i].neurons[j].bias+" ");
			}
		}
		
		outputs=mn.forwardPropagation(inputs);
		for(int i=0;i<outputs.length;i++){
			System.out.print(outputs[i]+" ");
		}
		System.out.println("\n\n\n\n\n");
		}
		
		*/
		
		//double[] target={1};
		
		//System.out.println(mn.costFunction(target));
	}

}
