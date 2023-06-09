package utils

// FIX THIS!
import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
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
	dataTensor := tensor.New(
		tensor.WithShape(len(data), 1),
		tensor.Of(tensor.Float64),
		tensor.WithBacking(data),
	)

	fmt.Println("LoadData.dataTensor:", dataTensor.Data())
	return dataTensor, nil
}

type singleIndexSlice struct {
	index  int
	data   interface{}
	stride int
	start  int
	end    int
}

func (s *singleIndexSlice) Start() int {
	return s.index
}

func (s *singleIndexSlice) Step() int {
	return 1
}

func (s *singleIndexSlice) End() int {
	return s.end
}

func (s *singleIndexSlice) Next() {
	s.start += s.stride
	s.end += s.stride
}

func (s *singleIndexSlice) At(i int) interface{} {
	return reflect.ValueOf(s.data).Index(s.start + i*s.stride).Interface()
}

func (s *singleIndexSlice) Reset() {
	s.start = 0
	s.end = s.stride
}
func (s *singleIndexSlice) String() string {
	return fmt.Sprintf("Slice(%d, %d, 1)", s.index, s.index+1)
}

// SplitData splits the data tensor into training and testing sets based on the given ratio
func SplitData(data tensor.Tensor, trainRatio float64) (tensor.Tensor, tensor.Tensor, error) {
	// Get the number of samples in the data
	numSamples := data.Shape()[0]

	// Convert the data to []float32
	// dataFloat32 := make([]float32, numSamples)
	it := data.Iterator()
	// for i := 0; ; i++ {
	// 	index, err := it.Next()
	// 	if err != nil {
	// 		if err == io.EOF {
	// 			break
	// 		}
	// 		return nil, nil, err
	// 	}

	// 	value, err := data.At(index)
	// 	if err != nil {
	// 		return nil, nil, err
	// 	}
	// 	dataFloat32[i] = value.(float32)
	// }

	// Convert the data to a tensor
	// TODO: FIX THIS!!!!
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
	trainData := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(shape...))

	// Populate training data
	for i := 0; i < numTrain; i++ {
		_, err := it.Next()
		if err != nil {
			return nil, nil, err
		}

		srcTrainData := tensor.New(tensor.Of(data.Dtype()), tensor.WithShape(numTrain))
		if err != nil {
			fmt.Println("data.Slice Train error:", err)
			return nil, nil, err
		}

		dstTrainData := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(shape...))
		if err := tensor.Copy(dstTrainData, srcTrainData); err != nil {
			fmt.Println("tensor.Copy error:", err)
			return nil, nil, err
		}

		for j := 0; j < dstTrainData.Shape()[0]; j++ {
			value, err := dstTrainData.At(j, 0) // Access element at index j
			if err != nil {
				return nil, nil, err
			}
			floatValue := value.(float64)
			dstTrainData.SetAt(floatValue+1, j, 0) // Assign modified value back to the tensor
		}

		if err := tensor.Copy(dstTrainData, srcTrainData); err != nil {
			fmt.Println("tensor.Copy error:", err)
			return nil, nil, err
		}
	}

	// Update the shape for testing data
	shape[0] = numTest

	// Create a new tensor for testing data
	testData := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(shape...))

	// Populate testing data
	for i := 0; i < numTest; i++ {
		index := indices[numTrain+i] // Use the correct index from the shuffled indices
		_, err := it.Next()
		if err != nil {
			return nil, nil, err
		}

		srcSlice := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1))
		srcSlice.Set(0, float64(index))

		srcTestData, err := srcSlice.Slice()
		if err != nil {
			fmt.Println("tensor.NewViewFrom error:", err)
			return nil, nil, err
		}

		dstTestData := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(shape...))
		if err := tensor.Copy(dstTestData, srcTestData); err != nil {
			fmt.Println("tensor.Copy error:", err)
			return nil, nil, err
		}

		for j := 0; j < dstTestData.Shape()[0]; j++ {
			value, err := dstTestData.At(j, 0) // Access element at index j
			if err != nil {
				return nil, nil, err
			}
			dstTestData.SetAt(value.(float64)+1, j, 0) // Assign modified value back to the tensor
		}

		testSlice := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1))
		if err != nil {
			return nil, nil, err
		}

		if err = tensor.Copy(testSlice, dstTestData); err != nil {
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
