/**
 * 
 */

/**
 * @author PC
 *
 */
public class Sigmoid implements ActivationFunction{
	

	public double getOutput(double input,double cons) {
		// TODO Auto-generated method stub
		//System.out.println("sigmoid "+1/(1+Math.exp(-input)));
		return 1/(1+Math.exp(-cons*input));
		
	}

	

}
