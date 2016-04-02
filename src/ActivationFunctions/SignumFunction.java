package ActivationFunctions;

public class SignumFunction implements ActivationFunction {

    private double alpha = 1;
    public SignumFunction(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double activationFunction(double S) {
        return Math.signum(S);
    }

    @Override
    public double functionDerivative(double S) {
        return 0;
    }
}
