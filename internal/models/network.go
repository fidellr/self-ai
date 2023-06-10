package models

import (
	"errors"
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Model struct {
	Graph                                        *gorgonia.ExprGraph
	Pred, Loss, X, Y, Wxh, Whh, Why, Cost, Grads *gorgonia.Node
	HiddenSize                                   int // Number of hidden units
}

const (
	batchSize   = 64
	hiddenSize  = 128
	inputSize   = 28
	outputSize  = 10
	sequenceLen = 28
)

func NewModel(g *gorgonia.ExprGraph) *Model {
	X := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(batchSize, inputSize))
	Y := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(batchSize, outputSize))
	Wxh := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(inputSize, hiddenSize))
	Whh := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(hiddenSize, hiddenSize))
	Why := gorgonia.NewMatrix(g, gorgonia.Float32, gorgonia.WithShape(hiddenSize, outputSize))

	return &Model{
		X:   X,
		Y:   Y,
		Wxh: Wxh,
		Whh: Whh,
		Why: Why,
	}
}

func (m *Model) rangeNodes(start, end int) []*gorgonia.Node {
	nodes := make([]*gorgonia.Node, end-start)
	for i := start; i < end; i++ {
		node := gorgonia.NewScalar(m.Graph, gorgonia.Float64, gorgonia.WithName(fmt.Sprintf("index_%d", i)), gorgonia.WithValue(float64(i)))
		nodes[i-start] = node
	}
	return nodes
}

func (m *Model) nodeToTensor(n *gorgonia.Node) (tensor.Tensor, error) {
	// Get the value of the node
	value := n.Value()

	// Convert the value to a tensor.Tensor
	t, ok := value.(tensor.Tensor)
	if !ok {
		return nil, fmt.Errorf("failed to convert *gorgonia.Node to tensor.Tensor")
	}

	return t, nil
}

// forwardPass performs the forward pass of the RNN model
func (m *Model) forwardPass(input *gorgonia.Node) (*gorgonia.Node, error) {
	// Perform the forward pass computation using the graph
	seqLen := m.X.Shape()[1]
	batchSize := m.X.Shape()[0]

	// Initialize the hidden state
	hPrev := gorgonia.NewMatrix(m.Graph, gorgonia.Float64, gorgonia.WithShape(batchSize, m.HiddenSize), gorgonia.WithName("hPrev"), gorgonia.WithInit(gorgonia.Zeroes()))

	// Store the hidden states for each time step
	var hiddenStates []tensor.Tensor

	// Iterate over the sequence length
	for t := 0; t < seqLen; t++ {
		// Retrieve the input at time step t
		x := gorgonia.Must(gorgonia.Slice(m.X, gorgonia.S(t)))

		// Concatenate the input with the previous hidden state
		concat := gorgonia.Must(gorgonia.Concat(1, x, hPrev))

		// Calculate the current hidden state
		h := gorgonia.Must(gorgonia.Mul(concat, m.Wxh))
		h = gorgonia.Must(gorgonia.Sigmoid(h))

		// Update the previous hidden state for the next time step
		hPrev = h

		// Store the current hidden state
		tensorH, err := m.nodeToTensor(h)
		if err != nil {
			log.Fatalln("m.nodeToTensor:", err)
			return nil, err
		}
		hiddenStates = append(hiddenStates, tensorH)
	}

	tStack, err := tensor.Stack(0, hiddenStates[0], hiddenStates[1:]...)
	if err != nil {
		log.Fatalln("tensor.Stack:", err)
		return nil, err
	}
	// Stack the hidden states along the time axis
	output := gorgonia.NodeFromAny(m.Graph, tStack, gorgonia.WithName("output"))

	return output, nil
}

// DefineNetwork defines your neural network architecture using Gorgonia's ExprGraph
func (m *Model) DefineNetwork() error {
	g := gorgonia.NewGraph()

	// Define input and target placeholders
	m.X = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(batchSize, inputSize), gorgonia.WithName("X"))
	m.Y = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(batchSize, outputSize), gorgonia.WithName("Y"))

	// Define the weights and biases
	m.Wxh = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(inputSize, hiddenSize), gorgonia.WithName("Wxh"))
	m.Whh = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, hiddenSize), gorgonia.WithName("Whh"))
	m.Why = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, outputSize), gorgonia.WithName("Why"))

	// Define the forward pass
	h := gorgonia.Must(gorgonia.Mul(m.X, m.Wxh))
	hShape := tensor.Shape{batchSize, inputSize, hiddenSize}
	h = gorgonia.Must(gorgonia.Reshape(h, hShape))

	for t := 0; t < sequenceLen; t++ {
		ht := gorgonia.Must(gorgonia.Slice(h, gorgonia.S(t*hiddenSize), gorgonia.S((t+1)*hiddenSize)))
		hh := gorgonia.Must(gorgonia.Mul(ht, m.Whh))
		h = gorgonia.Must(gorgonia.Add(h, hh))
	}

	out := gorgonia.Must(gorgonia.Mul(h, m.Why))
	outShape := tensor.Shape{batchSize, hiddenSize * outputSize}
	out = gorgonia.Must(gorgonia.Reshape(out, outShape))

	// Define the loss function and cost
	diff := gorgonia.Must(gorgonia.Sub(out, m.Y))
	cost := gorgonia.Must(gorgonia.Mul(diff, diff))
	m.Cost = gorgonia.Must(gorgonia.Mean(cost))

	// Define the gradients and gradients' ops
	m.Grads = []*gorgonia.Node{m.Wxh, m.Whh, m.Why}[0]

	return nil
}

// inputShape returns the shape of the input tensor
func (m *Model) inputShape() []int {
	// Return the shape of the input tensor
	return []int{batchSize, inputSize}
}

func (m *Model) inputTensor(g *gorgonia.ExprGraph, data []float64, shape []int) (*gorgonia.Node, error) {
	expectedSize := shape[0]

	if len(data) != expectedSize {
		return nil, fmt.Errorf("inputTensor: data size doesn't match the specified shape")
	}

	t := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))
	input := gorgonia.NewTensor(g, t.Dtype(), t.Dims(), gorgonia.WithShape(t.Shape()...), gorgonia.WithValue(t))
	err := gorgonia.Let(input, t)
	if err != nil {
		return nil, fmt.Errorf("inputTensor: failed to assign value to tensor")
	}

	return input, nil
}

func (m *Model) outputTensor(g *gorgonia.ExprGraph, data []float64, shape []int) *gorgonia.Node {
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

// gradWrapper is a custom implementation of the gorgonia.ValueGrad interface
type gradWrapper struct {
	*gorgonia.Node
}

type gradValue struct {
	value gorgonia.Value
}

func newGradValue(value gorgonia.Value) gorgonia.Value {
	return &gradValue{value}
}

func (gw *gradWrapper) Grad() (gorgonia.Value, error) {
	grad, err := gorgonia.Grad(gw.Node)
	if err != nil {
		return nil, err
	}
	return grad.Node().Value(), nil
}
func (gw *gradWrapper) Data() gorgonia.Value {
	return gw.Node.Value()
}
func (gw *gradWrapper) MemSize() uintptr {
	return gw.Value().MemSize()
}

func (gv *gradValue) Dtype() tensor.Dtype {
	return gv.value.Dtype()
}
func (gv *gradValue) Shape() tensor.Shape {
	return gv.value.Shape()
}
func (gv *gradValue) Size() int {
	return gv.value.Size()
}
func (gv *gradValue) ScalarValue() interface{} {
	return gv.value.Data()
}
func (gv *gradValue) Format(state fmt.State, c rune) {
	gv.value.Format(state, c)
}
func (gv *gradValue) Grad() (gorgonia.Value, error) {
	return nil, fmt.Errorf("Grad() not implemented for gradValue")
}
func (gv *gradValue) Data() interface{} {
	return gv.value.Data()
}
func (gv *gradValue) MemSize() uintptr {
	return gv.value.MemSize()
}
func (gv *gradValue) Uintptr() uintptr {
	return gv.value.Uintptr()
}
func (gv *gradValue) Clone() gorgonia.Value {
	clone, _ := gorgonia.CloneValue(gv.value)
	return &gradValue{value: clone}
}
func (gv *gradValue) Reshape(...int) (tensor.Tensor, error) {
	return nil, errors.New("Reshape is not supported for gradValue")
}
func (gv *gradValue) ReshapeDense(...int) (tensor.Tensor, error) {
	return nil, errors.New("ReshapeDense is not supported for gradValue")
}

// Train trains the model using the provided training data
func (m *Model) Train(g *gorgonia.ExprGraph, data [][]float64, labels [][]float64, epochs int, learningRate float64) error {

	// Define the optimization algorithm
	optimizer := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(learningRate))

	// Define the forward pass and loss function
	m.DefineNetwork()

	// Get the model parameters
	params := []*gorgonia.Node{m.Wxh}

	// Perform training
	for epoch := 0; epoch < epochs; epoch++ {
		// Iterate over the training data
		for i := range data {
			// Create input tensor
			dims := m.inputShape()[0] * m.inputShape()[1]
			inputTensor := gorgonia.NewTensor(g, gorgonia.Float64, dims, gorgonia.WithShape(m.inputShape()...), gorgonia.WithValue(tensor.WithBacking(data)))

			// Forward pass and compute gradients
			// Forward pass
			// Forward pass
			output, err := m.forwardPass(inputTensor)
			if err != nil {
				return err
			}

			// Convert labels to a *gorgonia.Node
			labelTensor := gorgonia.NewTensor(g, gorgonia.Float64, dims, gorgonia.WithShape(m.inputShape()...), gorgonia.WithValue(tensor.WithBacking(labels[i])))

			// Calculate loss (squared distance)
			diff := gorgonia.Must(gorgonia.Sub(output, labelTensor))
			lossFunc := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(diff))))

			// Calculate gradients
			grads, err := gorgonia.Grad(lossFunc, params...)
			if err != nil {
				return err
			}

			fmt.Println("m.Grads: ", m.Grads, "Grads:", grads)
			// Convert grads from Nodes to ValueGrads
			gradValues := make([]gorgonia.ValueGrad, grads.Len())
			for i := 0; i < grads.Len(); i++ {
				val := newGradValue(grads[i].Value())
				gradValues[i] = val.Data().(gorgonia.ValueGrad)
			}

			// Update weights and biases using the computed gradients
			if err := optimizer.Step(gradValues); err != nil {
				return err
			}
		}
	}

	return nil
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
