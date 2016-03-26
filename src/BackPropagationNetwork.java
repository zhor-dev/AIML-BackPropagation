import java.io.*;

public class BackPropagationNetwork extends Network {

    protected double epsilon;
    private double [][]oldWeight;
    private double [][]oldWeight1;
    private double [][]currWeight;
    private double BETTA = 0.1;
    private double GAMMA = 0.0001;
    private double [][]desOutputs;
    private double momentum;
    private boolean momentumChanged = false;
    private boolean updateCycle = false;

    public BackPropagationNetwork(double []i, double [][]dOutput, int []lSize, int dOutputIndex, ActivationFunction af) {
        super(dOutput[dOutputIndex], i, lSize, af);
        this.desOutputs = dOutput;
        setOldWeight(oldWeight);
        setOldWeight(oldWeight1);
    }

    public BackPropagationNetwork(int inputSize, int []lSize, double [][]dOutput, ActivationFunction af) {
        super(inputSize, lSize, af);
        this.desOutputs = dOutput;
    }

    public void disableMomentum() {
        BETTA = 0;
    }

    public void disableWeightMinimization() {
        GAMMA = 0;
    }

    private double[][] setOldWeight(double [][]w) {
        w = new double[layersSize.length][];
        for (int j = 0; j < layersSize.length; ++j) {
            w[j] = new double[layersSize[j] * layers[j].getNeurons()[0].getWeights().length];
        }
        return w;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setAlpha(double a) {
        ActivationFunctions.alpha = a;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
        momentumChanged = true;
    }

    public void trainNetwork() {
        if (oldWeight1 == null) {
            oldWeight = setOldWeight(oldWeight);
            oldWeight1 = setOldWeight(oldWeight1);
            currWeight = setOldWeight(currWeight);
        }
        double e = 0;
        for (int i = 0; i < layers[layers.length - 1].getNeurons().length; ++i) {
            e += Math.pow(networkOutput()[i] - desiredOutput[i], 2);
        }
       //if (!isCorrect(e)) {
            updateCycle = !updateCycle;
            if (updateCycle) {
                updateOldWeights(oldWeight);
                currWeight = oldWeight1;
            } else {
                updateOldWeights(oldWeight1);
                currWeight = oldWeight;
            }
            updateWeights(epsilon);
        //}
    }

    public void updateWeights(double e) {
        for (int i = 0; i < layersSize[layersSize.length - 1]; ++i) {
            for (int j = 0; j < layersSize[layersSize.length - 2]; ++j) {
                layers[layers.length - 1].getNeurons()[i].getWeights()[j] =
                        (1 - GAMMA) * layers[layers.length - 1].getNeurons()[i].getWeights()[j] -
                        e * 2 * (layers[layers.length - 1].getNeurons()[i].neuronOutput() - desiredOutput[i]) *
                        function.functionDerivative(layers[layers.length - 1].getNeurons()[i].summingBlock()) *
                        layers[layers.length - 2].getNeurons()[j].neuronOutput() + BETTA *
                        momentum(layers[layers.length - 1].getNeurons()[i].getWeights()[j], layers.length - 1, i, j);
            }
        }
        int k = layers.length - 2;
        double []delta = new double[layers[k].getNeurons().length];
        for (int i = 0; i < layers[k].getNeurons().length; ++i) {
            double sum = 0;
            for (int t = 0; t < layers[k + 1].getNeurons().length; ++t) {
                sum += (layers[k + 1].layerOutput()[t] - desiredOutput[t]) *
                        function.functionDerivative(layers[k + 1].getNeurons()[t].neuronOutput()) *
                        layers[k + 1].getNeurons()[t].getWeights()[i];
            }
            delta[i] = sum;
            for (int j = 0; j < layers[k].getNeurons()[0].getInputs().length; ++j) {
                layers[k].getNeurons()[i].getWeights()[j] =
                        (1 - GAMMA) * layers[k].getNeurons()[i].getWeights()[j] -
                        e * 2 * function.functionDerivative(layers[k].getNeurons()[i].summingBlock()) *
                        layers[k].getNeurons()[i].getInputs()[j] * delta[i] + BETTA *
                        momentum(layers[k].getNeurons()[i].getWeights()[j], k, i, j) ;
            }
        }
        for (int r = k - 1; r >= 0; --r) {
            double []delta1 = new double[layers[r].getNeurons().length];
            for (int i = 0; i < layers[r].getNeurons().length; ++i) {
                double sum = 0;
                for (int t = 0; t < layers[r + 1].getNeurons().length; ++t) {
                    sum += delta[t] *
                           function.functionDerivative(layers[r + 1].getNeurons()[t].neuronOutput()) *
                           layers[r + 1].getNeurons()[t].getWeights()[i];
                }
                delta1[i] = sum;
                for (int j = 0; j < layers[r].getNeurons()[0].getInputs().length; ++j) {
                    layers[r].getNeurons()[i].getWeights()[j] =
                            (1 - GAMMA) * layers[r].getNeurons()[i].getWeights()[j] -
                            e * 2 * function.functionDerivative(layers[r].getNeurons()[i].summingBlock()) *
                            layers[r].getNeurons()[i].getInputs()[j] * delta1[i] + BETTA *
                            momentum(layers[r].getNeurons()[i].getWeights()[j], r, i, j) ;
                }
            }
            delta = delta1;
        }
    }

    private void updateOldWeights(double [][]w) {
        for (int j = 0; j < layers.length; ++j) {
            int index = 0;
            for (int t = 0; t < layers[j].getNeurons().length; ++t) {
                for (int k = 0; k < layers[j].getNeurons()[t].getWeights().length; ++k) {
                    w[j][index + k] = layers[j].getNeurons()[t].getWeights()[k];
                }
                index += layers[j].getNeurons()[t].getWeights().length;
            }
        }
    }

    private double momentum(double currentWeight, int indexLayer, int indexNeuron, int indexWeight) {
        if (!momentumChanged) {
            return currentWeight - currWeight[indexLayer][indexNeuron *
                    layers[indexLayer].getNeurons()[0].getWeights().length + indexWeight];
        } else {
            return momentum;
        }
    }

    private boolean isCorrect(double e) {
        double err = 0;
        for (int i = 0; i < desOutputs[0].length; ++i) {
            double []outputs = desOutputs[i];
            if (notCurrentOutput(outputs)) {
                for (int j = 0; j < layersSize[layersSize.length - 1]; ++j) {
                    err += Math.pow(layers[layers.length - 1].layerOutput()[j] - outputs[j], 2);
                }
                if (err < e) {
                    return false;
                }
            }
            err = 0;
        }
        return true;
    }

    private boolean notCurrentOutput(double []output) {
        for (int i = 0; i < output.length; ++i) {
            if (output[i] != desiredOutput[i]) {
                return true;
            }
        }
        return false;
    }

    public void saveWeights() {
        String res = "";
        BufferedWriter output = null;
        try {
            File file = new File("src/weights.txt");
            output = new BufferedWriter(new FileWriter(file));
            for (int k = layers.length - 1; k >= 1; --k) {
                for (int i = 0; i < layers[k].getNeurons().length; ++i) {
                    for (int j = 0; j < layers[k - 1].getNeurons().length; ++j) {//+1
                        res = res + layers[k].getNeurons()[i].getWeights()[j] + "\n";
                    }
                }
            }
            for (int r = 0; r < layers[0].getNeurons().length; ++r) {
                for (int s = 0; s < layers[0].getNeurons()[0].getInputs().length; ++s) {//+1
                    res = res + layers[0].getNeurons()[r].getWeights()[s] + "\n";
                }
            }
            output.write(res);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (output != null) try {
                output.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static double[][][] loadWeights(double [][][]weights) {
        BufferedReader buffreader = null;
        try {
            FileReader fileReader = new FileReader("src/weights.txt");
            buffreader = new BufferedReader(fileReader);
            for (int k = weights.length - 1; k >= 0; --k) {
                for (int i = 0; i < weights[k].length; ++i) {
                    for (int j = 0; j < weights[k][0].length; ++j) {
                        weights[k][i][j] = Double.parseDouble(buffreader.readLine());
                    }
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        } finally {
            try {
                assert buffreader != null;
                buffreader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return weights;
    }

}
