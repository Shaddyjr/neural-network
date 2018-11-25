class NeuralNetwork{
    static get DEFAULT_LEARNING_RATE(){
        return .1;
    }
    
    constructor(inputs, hidden, output, {learningRate}={}){
        this.hidden = Array.isArray(hidden) ? hidden : [hidden]; // array of numbers representing # of nodes in a layer
        if(this.hidden.some(val=>val < 1)) throw new Error("Cannot have less than 1 node in hidden layer");
        this.num_of_inputs = inputs;
        this.num_of_output = output;
        this.learningRate = learningRate || NeuralNetwork.DEFAULT_LEARNING_RATE;
        this.initWeights();
        this.initBiases();
    }

    initWeights(){
        // must set up weight matrices between layers
        // Matrix object automatically adds random values to new matrix
        this.weights = [];

        // FIRST matrix is input to hidden
        const input_to_hidden = new Matrix(this.hidden[0], this.num_of_inputs);
        this.weights.push(input_to_hidden);

        const lastHiddenIndex = this.hidden.length-1;
        for(let i = 1; i <= lastHiddenIndex; i++){
            const numNodesInHidden = this.hidden[i];
            const numNodesInLast = this.hidden[i-1];
            const hidden_layer = new Matrix(numNodesInHidden, numNodesInLast);
            this.weights.push(hidden_layer);
        }

        // LAST matrix is hidden to output
        const hidden_to_output = new Matrix(this.num_of_output, this.hidden[lastHiddenIndex]);
        this.weights.push(hidden_to_output);
    }
    
    initBiases(){
        // must set biases for each layer
        this.biases = []; 
        
        for(let i = 0; i < this.hidden.length; i++){
            const numNodesInHidden = this.hidden[i];
            const hidden_layer = new Matrix(numNodesInHidden, 1);
            this.biases.push(hidden_layer);
        }

        // LAST is hidden to output
        const hidden_to_output_bias = new Matrix(this.num_of_output,1);
        this.biases.push(hidden_to_output_bias);
    }

    /**
     * Returns input number converted into value between -1 and 1.
     */
    static sigmoid(n){
        return 1 / (1 + Math.exp(-n));
    }

    static derivative_sigmoid(x){
        // const val = NeuralNetwork.sigmoid(x);
        // return val * (1-val);
        return x * (1-x); // hack since variable already "sigmoided"
    }

    feedforward(input_arr){
        let input = Matrix.fromArray(input_arr);

        const l = this.weights.length;
        for(let i = 0; i < l; i++){
            const current = this.weights[i];
            const currentBias = this.biases[i];
            input = Matrix.multiply(current, input);
            input.add(currentBias);
            input.map(NeuralNetwork.sigmoid);
        }

        return input.toArray();
    }

    train(input_arr, actuals_arr){
        // Need to retain data for backpropogation
        let input = Matrix.fromArray(input_arr);        
        const trackedOutputs = [input]; // essentially first "output" from upstream
        const l = this.weights.length;
        for(let i = 0; i < l; i++){
            const current = this.weights[i];
            const currentBias = this.biases[i];
            input = Matrix.multiply(current, input);
            trackedOutputs.push(input); // values coming out of this layer
            input.add(currentBias);
            input.map(NeuralNetwork.sigmoid);
        }
        
        // OUTPUT LAYER
        // Output Error = targets - outputs
        const actuals = Matrix.fromArray(actuals_arr);
        let output_errors = Matrix.subtract(actuals, trackedOutputs[trackedOutputs.length-1]);
        // Looping backwards, starting from 2nd to last b/c of special treatment for how last error calculated
        for(let i = this.weights.length-1; i >= 0; i--){
            // CALCULATING GRADIENT FOR CURRENT LAYER
            const current_output = trackedOutputs[i+1]; // current layer's output is downstream of it's index
            const gradients = Matrix.map(current_output, NeuralNetwork.derivative_sigmoid);        
            gradients.multiply(output_errors);
            gradients.multiply(this.learningRate);

            // CALCULATING DELTAS
            const previous_output = trackedOutputs[i]; // !!!?
            const T_previous_output = Matrix.transpose(previous_output);
            const deltas = Matrix.multiply(gradients, T_previous_output);

            // // TWEEKING WEIGHTS
            const currentWeight = this.weights[i];
            const currentBias = this.biases[i];
            currentWeight.add(deltas);
            currentBias.add(gradients);

            // // SETTING UP NEXT ERROR
            const T_currentWeight = Matrix.transpose(currentWeight);
            output_errors = Matrix.multiply(T_currentWeight, output_errors);
        }
    }
}