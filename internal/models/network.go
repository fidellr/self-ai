package models

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Model struct {
	Graph *gorgonia.ExprGraph
	Pred  *gorgonia.Node
	Loss  *gorgonia.Node
	W     *gorgonia.Node
	B     *gorgonia.Node
	X     *gorgonia.Node
	Y     *gorgonia.Node
}

// DefineNetwork defines your neural network architecture using Gorgonia's ExprGraph
func DefineNetwork() *Model {
	// Create the graph
	g := gorgonia.NewGraph()

	// Define your model parameters
	w := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, 1), gorgonia.WithName("w"), gorgonia.WithInit(gorgonia.Gaussian(0, 1)))
	b := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("b"), gorgonia.WithInit(gorgonia.Zeroes()))

	// Define your input and output tensors
	x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, 1), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, 1), gorgonia.WithName("y"))

	// Define your model
	pred := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, w)), b))

	// Define the loss function
	loss := gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(pred, y))))

	return &Model{g, pred, loss, w, b, x, y}
}

func (m *Model) inputTensor(g *gorgonia.ExprGraph, data []float64, shape ...int) (*gorgonia.Node, error) {
	expectedSize := 1
	for _, dim := range shape {
		expectedSize *= dim
	}

	if len(data) != expectedSize {
		return nil, fmt.Errorf("inputTensor: data size doesn't match the specified shape")
	}

	t := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))
	input := gorgonia.NewTensor(g, t.Dtype(), t.Dims(), gorgonia.WithShape(t.Shape()...), gorgonia.WithName("input"))
	return input, nil
}

func (m *Model) outputTensor(g *gorgonia.ExprGraph, data []float64, shape ...int) *gorgonia.Node {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	if len(data) != size {
		log.Fatalln("data size does not match tensor shape")
	}

	dataT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))

	return gorgonia.NewTensor(g, dataT.Dtype(), len(shape), gorgonia.WithValue(dataT))
}

func (m *Model) computeAccuracy(predictions, targets []float64) float64 {
	correct := 0
	total := len(predictions)

	for i := 0; i < total; i++ {
		if predictions[i] == targets[i] {
			correct++
		}
	}

	accuracy := float64(correct) / float64(total)
	return accuracy
}

// Train trains the model using the provided training data
func (m *Model) Train(g *gorgonia.ExprGraph, data tensor.Tensor) {

	// Create a solver for gradient descent optimization
	learnRate := 0.1
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(learnRate))

	// Define the number of iterations for training
	numIterations := 100
	batchSize := 32 // Replace with your desired batch size
	inputSize := 10 // Replace with your desired input size

	// Define the shape
	shapes := []int{batchSize, inputSize}

	// Perform training
	for i := 0; i < numIterations; i++ {
		// Forward pass to compute the loss
		dataVals := data.Data().([]float64)
		input, err := m.inputTensor(g, dataVals, shapes...)

		if err != nil {
			log.Fatalf("input: %v", err)
		}
		if err = gorgonia.Let(m.X, input); err != nil {
			log.Fatal("inputTensor:", err)
		}
		if err := gorgonia.Let(m.Y, m.outputTensor(g, dataVals, shapes...)); err != nil {
			log.Fatal("outputTensor:", err)
		}

		// Calculate the gradients
		bGrad, err := m.B.Grad()
		if err != nil {
			// Handle the error appropriately
			fmt.Println("Error getting gradient for b:", err)
			return
		}
		wGrad, err := m.W.Grad()
		if err != nil {
			// Handle the error appropriately
			fmt.Println("Error getting gradient for w:", err)
			return
		}
		bGradShape := bGrad.Shape()
		wGradShape := wGrad.Shape()

		bGradNode := gorgonia.NewTensor(g, bGrad.Dtype(), bGradShape.Dims(), gorgonia.WithShape(bGradShape...), gorgonia.WithValue(bGrad))
		wGradNode := gorgonia.NewTensor(g, wGrad.Dtype(), wGradShape.Dims(), gorgonia.WithShape(wGradShape...), gorgonia.WithValue(wGrad))

		grads := []gorgonia.ValueGrad{wGradNode, bGradNode}
		// Update parameters
		if err := solver.Step(grads); err != nil {
			log.Fatal(err)
		}
	}

	// Print the final parameters
	wVal := m.W.Value()
	bVal := m.B.Value()
	// Calculate loss
	lossVal := m.Loss.Value().(*tensor.Dense).Data()
	// Assuming pred and y are both tensor.Dense
	predVal := m.Pred.Value().(*tensor.Dense).Data()
	yVal := m.Y.Value().(*tensor.Dense).Data()

	// Compute accuracy
	accuracy := m.computeAccuracy(predVal.([]float64), yVal.([]float64))

	fmt.Println("Training accuracy:", accuracy)
	fmt.Println("Current loss:", lossVal)
	fmt.Println("Final parameters:")
	fmt.Println("w =", wVal)
	fmt.Println("b =", bVal)
}

// Evaluate evaluates the model on the provided test data and returns the accuracy
func (model *Model) Evaluate(testData tensor.Tensor) float64 {
	// Implement the necessary code for model evaluation
	fmt.Println("Evaluating the model...")
	return 0.0
}

// Predict makes predictions with the trained model on the given sample data
func (model *Model) Predict(sampleData tensor.Tensor) float64 {
	// Implement the necessary code for making predictions
	fmt.Println("Making predictions...")
	return 0.0
}
