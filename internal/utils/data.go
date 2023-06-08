package utils

// FIX THIS!
import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gorgonia.org/tensor"
)

// LoadData loads and preprocesses EEG data from a .tsv file
func LoadData(filePath string) (tensor.Tensor, error) {
	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}

	// Construct the absolute file path
	filePath = filepath.Join(wd, filePath)

	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	// Parse the content
	lines := strings.Split(string(content), "\n")

	// Initialize the data slice
	var data []float64

	// Process the lines and construct the data slice
	for i, line := range lines {
		// Skip empty lines and the header line
		if line == "" || i == 0 {
			continue
		}

		// Split the line into individual values
		values := strings.Split(line, "\t")

		// Process each value and add it to the data slice
		for _, value := range values {
			// Skip non-numeric values
			if value == "" {
				continue
			}

			// Convert the value to float64 and add it to the data slice
			// Handle any necessary data preprocessing steps here
			floatValue, err := strconv.ParseFloat(value, 64)
			if err != nil {
				// Skip non-float values and try parsing as integer
				intValue, err := strconv.Atoi(value)
				if err != nil {
					continue
				}
				data = append(data, float64(intValue))
			} else {
				data = append(data, floatValue)
			}
		}
	}

	// Create a tensor from the data slice
	dataTensor := tensor.New(tensor.WithShape(len(data)), tensor.WithBacking(data))

	fmt.Println(dataTensor)
	return dataTensor, nil
}

type singleIndexSlice struct {
	index int
}

func (s singleIndexSlice) Start() int {
	return s.index
}

func (s singleIndexSlice) End() int {
	return s.index + 1
}

func (s singleIndexSlice) Step() int {
	return 1
}

// SplitData splits the data tensor into training and testing sets based on the given ratio
func SplitData(data tensor.Tensor, trainRatio float64) (tensor.Tensor, tensor.Tensor, error) {
	// Get the number of samples in the data
	numSamples := data.Shape()[0]

	// Calculate the number of samples for training and testing
	numTrain := int(float64(numSamples) * trainRatio)
	numTest := numSamples - numTrain

	// Create a range of indices for shuffling
	indices := make([]int, numSamples)
	for i := 0; i < numSamples; i++ {
		indices[i] = i
	}
	rand.Shuffle(numSamples, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

	// Get the shape of the data tensor
	shape := data.Shape()
	shape[0] = numTrain

	// Create a new tensor for training data
	trainData := tensor.New(tensor.WithShape(shape...), tensor.Of(data.Dtype()))

	// Populate training data
	for i := 0; i < numTrain; i++ {
		dstTrainSlice, err := trainData.Slice(singleIndexSlice{index: indices[i]})
		if err != nil {
			fmt.Println("dstTrainSlice:", err, dstTrainSlice, numTrain)
			return nil, nil, err
		}
		srcTrainSlice, err := data.Slice(singleIndexSlice{index: indices[i]})
		if err != nil {
			fmt.Println("srcTrainSlice:", err)
			return nil, nil, err
		}
		if err = tensor.Copy(dstTrainSlice, srcTrainSlice); err != nil {
			fmt.Println("copyTrain:", err)
			return nil, nil, err
		}
	}

	// Update the shape for testing data
	shape[0] = numTest

	// Create a new tensor for testing data
	testData := tensor.New(tensor.WithShape(shape...), tensor.Of(data.Dtype()))

	// Populate testing data
	for i := 0; i < numTest; i++ {
		dstTestSlice, err := testData.Slice(singleIndexSlice{index: indices[i+numTrain]})
		if err != nil {
			fmt.Println("dstTestSlice:", err)
			return nil, nil, err
		}
		srcTestSlice, err := data.Slice(singleIndexSlice{index: indices[i+numTrain]})
		if err != nil {
			fmt.Println("srcTestSlice:", err)
			return nil, nil, err
		}
		if err = tensor.Copy(dstTestSlice, srcTestSlice); err != nil {
			fmt.Println("copyTest:", err)
			return nil, nil, err
		}
	}

	return trainData, testData, nil
}

// ShuffleData shuffles the data tensor randomly
func ShuffleData(data tensor.Tensor) {
	// Get the number of samples in the data tensor
	numSamples := data.Shape()[0]

	// Generate a random permutation of sample indices
	perm := rand.Perm(numSamples)

	// Convert the data tensor to a flat array
	dataArray := data.Data().([]float64)

	// Permute the underlying data array based on the generated indices
	for i, j := range perm {
		dataArray[i], dataArray[j] = dataArray[j], dataArray[i]
	}
}

// GetSampleData returns sample data for making predictions
func GetSampleData() tensor.Tensor {
	// Implement the necessary code to get sample data for predictions
	fmt.Println("Getting sample data...")
	return nil
}
